"""RUT — Rank-Based Uniformity Test (Zhu et al., ICLR 2026, arXiv 2506.06975), avec les vrais checkpoints locaux.

Pour chaque requête, la réponse de l'endpoint est notée par sa log-vraisemblance sous le modèle de référence (variante v,
température T_REF) ; on la situe parmi les notes de M échantillons locaux tirés du même modèle de référence pour le même
prompt. Sous H0 (endpoint = référence), le rang normalisé est uniforme sur [0, 1] ; test de Cramér-von Mises.
On répète contre chaque variante de référence (bf16, fp16, fp8w, int8w) : la variante non rejetée / au plus grand p
est la quantization attribuée.

    python sim/rut.py --pool sim/out/lik3 --api sim/out/lik_r2 --fam LL1 --eps ep-05,ep-09 [--prefix v1_,v2_]
Fichiers : pool = <modèle>__<v>__all_pool_<v>.jsonl (endpoint <FAM>-pool_<v>) ; api = <modèle>__<v>__(eps|refs|all...).jsonl
"""
import argparse, collections, glob, json, math, os, random
import numpy as np
from scipy import stats

TEMPS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
BASE = {"LL1": "unsloth__Llama-3.2-1B-Instruct", "QA3": "Qwen__Qwen2.5-3B-Instruct",
        "QB15": "Qwen__Qwen2.5-1.5B-Instruct", "S360": "HuggingFaceTB__SmolLM2-360M-Instruct"}


def load_scores(path, T, eps=None, per_token=False):
    i = TEMPS.index(T)
    out = collections.defaultdict(list)          # (endpoint, prompt_id) -> [(repeat, score)]
    for l in open(path):
        r = json.loads(l)
        if eps and r["endpoint"] not in eps:
            continue
        s = -r["nll"][i] / (r["n"] if per_token else 1)
        out[(r["endpoint"], r["prompt_id"])].append((r["repeat"], s))
    return out


def rut_one(api, pool, rng):
    """api : {prompt_id: [scores]} ; pool : {prompt_id: [scores]} -> rangs normalisés, stat CvM, p"""
    us = []
    for pid, ss in api.items():
        ref = pool.get(pid)
        if not ref or len(ref) < 8:
            continue
        ref = np.asarray(ref)
        for s in ss:
            lo = (ref < s).sum(); eq = (ref == s).sum()
            us.append((lo + rng.random() * (eq + 1)) / (len(ref) + 1))
    if len(us) < 5:
        return None
    us = np.asarray(us)
    r = stats.cramervonmises(us, "uniform")
    return {"n": len(us), "cvm": float(r.statistic), "p": float(r.pvalue), "mean_u": float(us.mean()),
            "ks_p": float(stats.kstest(us, "uniform").pvalue)}


def run(pool_dir, api_files, fam, eps, T=0.7, variants=("bf16", "fp16", "fp8w", "int8w"), prefix=None, per_token=False, seed=0):
    """api_files : {variante: chemin du fichier de notes des réponses de l'endpoint sous cette variante}"""
    rng = random.Random(seed)
    res = collections.defaultdict(dict)
    for v in variants:
        pp = os.path.join(pool_dir, f"{BASE[fam]}__{v}__all_pool_{v}.jsonl")
        if not os.path.exists(pp) or v not in api_files or not os.path.exists(api_files[v]):
            continue
        pool = collections.defaultdict(list)
        for (e, pid), lst in load_scores(pp, T, per_token=per_token).items():
            if e == f"{fam}-pool_{v}":
                pool[pid] += [s for _, s in lst]
        api = load_scores(api_files[v], T, eps=set(eps) if eps else None, per_token=per_token)
        by_ep = collections.defaultdict(dict)
        for (e, pid), lst in api.items():
            if prefix and not pid.startswith(tuple(prefix)):
                continue
            by_ep[e][pid] = [s for _, s in lst]
        for e, d in by_ep.items():
            r = rut_one(d, pool, rng)
            if r:
                res[e][v] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sim/out/lik3")
    ap.add_argument("--api", default="sim/out/lik_r2")
    ap.add_argument("--fam", required=True)
    ap.add_argument("--records", default="eps")
    ap.add_argument("--eps", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--T", type=float, default=0.7)
    ap.add_argument("--per-token", action="store_true")
    a = ap.parse_args()
    vs = ("bf16", "fp16", "fp8w", "int8w")
    api_files = {v: os.path.join(a.api, f"{BASE[a.fam]}__{v}__{a.records}.jsonl") for v in vs}
    if not os.path.exists(api_files["fp16"]):
        api_files["fp16"] = api_files["bf16"]       # fp16 et bf16 donnent les mêmes notes à 1e-3 près
    res = run(a.pool, api_files, a.fam, a.eps.split(",") if a.eps else None, a.T, vs, a.prefix.split(",") if a.prefix else None, a.per_token)
    print(f"### RUT famille {a.fam} — p de Cramér-von Mises [rang moyen] par variante de référence ; * = meilleure")
    print(f"{'endpoint':10s} {'n':>4s} " + " ".join(f"{v:>16s}" for v in vs))
    for e in sorted(res):
        best = max(res[e], key=lambda v: res[e][v]["p"])
        print(f"{e:10s} {next(iter(res[e].values()))['n']:4d} " + " ".join(
            (f"{res[e][v]['p']:9.2g} [{res[e][v]['mean_u']:.2f}]" + ("*" if v == best else " ")) if v in res[e] else f"{'-':>16s}" for v in vs))


if __name__ == "__main__":
    main()
