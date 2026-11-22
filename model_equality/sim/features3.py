"""Traits par « instance d'endpoint » (jeu n° 3 simulé, ou vrais endpoints), identiques des deux côtés.
Une instance = un endpoint x un bloc de réponses (repeat // 1000), restreinte aux requêtes d'un budget donné.

    from features3 import build ; X = build(lik_dir, pool_dir, fam, records_tag, srcs)   -> {(endpoint, bloc): {trait: valeur}}
"""
import collections, json, math, os, random
import numpy as np
from scipy import stats

TEMPS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
I7 = TEMPS.index(0.7)
BASE = {"LL1": "unsloth__Llama-3.2-1B-Instruct", "QA3": "Qwen__Qwen2.5-3B-Instruct",
        "QB15": "Qwen__Qwen2.5-1.5B-Instruct", "S360": "HuggingFaceTB__SmolLM2-360M-Instruct"}
TAGS = {"LL1": "ll1", "QA3": "q3", "QB15": "q15", "S360": "s360"}
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {json.loads(l)["prompt_id"]: json.loads(l)["src"] for l in open(os.path.join(HERE, "protocol2.jsonl"), encoding="utf-8")}
PROTO_KEYS = collections.defaultdict(set)      # src -> {(prompt_id, repeat)}
for l in open(os.path.join(HERE, "protocol2.jsonl"), encoding="utf-8"):
    j = json.loads(l); PROTO_KEYS[j["src"]].add((j["prompt_id"], j["repeat"]))


def _load(path, eps=None):
    d = {}
    if not os.path.exists(path):
        return d
    for l in open(path):
        r = json.loads(l)
        if eps is None or r["endpoint"] in eps:
            d[(r["endpoint"], r["prompt_id"], int(r["repeat"]))] = r
    return d


def _pool(pool_dir, fam, v):
    p = os.path.join(pool_dir, f"{BASE[fam]}__{v}__all_pool_{v}.jsonl")
    out = collections.defaultdict(list)
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            out[r["prompt_id"]].append(-r["nll"][I7])
    return {k: np.sort(np.asarray(v)) for k, v in out.items()}


def _llr(dq, recs):
    vals, ns = [], []
    for k, rb in recs:
        q = dq.get(k)
        if q is None or q["n"] != rb["n"]:
            continue
        vals.append(rb["nll"][I7] - q["nll"][I7]); ns.append(rb["n"])
    if len(vals) < 3:
        return float("nan"), float("nan")
    vals, ns = np.array(vals), np.array(ns)
    m = vals.sum() / ns.sum(); se = math.sqrt(((vals - m * ns) ** 2).sum()) / ns.sum()
    return m * 1000, (m / se if se > 0 else 0.0)


def _rut(recs, dv, pool, rng):
    us = []
    for k, rb in recs:
        r = dv.get(k) if dv is not None else rb
        ref = pool.get(k[1])
        if r is None or ref is None or len(ref) < 8:
            continue
        s = -r["nll"][I7]
        lo = np.searchsorted(ref, s, "left"); hi = np.searchsorted(ref, s, "right")
        us.append((lo + rng.random() * (hi - lo + 1)) / (len(ref) + 1))
    if len(us) < 8:
        return float("nan"), float("nan"), float("nan")
    us = np.asarray(us)
    c = stats.cramervonmises(us, "uniform")
    return float(us.mean()), float(c.statistic), float(-math.log10(max(c.pvalue, 1e-12)))


def build(lik_dir, pool_dir, fam, rec_tag, srcs=("v1", "v2a"), eps=None, blocks=True, proxy="alp3-1.0"):
    b = BASE[fam]
    f = lambda v: os.path.join(lik_dir, f"{b}__{v}__{rec_tag}.jsonl")
    db = _load(f("bf16"), eps)
    d8 = _load(f("fp8w"), eps); di = _load(f("int8w"), eps); dp = _load(f(f"lora-{TAGS[fam]}_{proxy}"), eps)
    pools = {v: _pool(pool_dir, fam, v) for v in ("bf16", "fp8w", "int8w")}
    keep = set().union(*[PROTO_KEYS[s] for s in srcs])
    inst = collections.defaultdict(list)
    for k, r in db.items():
        blk = k[2] // 1000 if blocks else 0
        if (k[1], k[2] % 1000 if blocks else k[2]) in keep:
            inst[(k[0], blk)].append((k, r))
    rng = random.Random(0)
    X = {}
    for key, recs in inst.items():
        if len(recs) < 20:
            continue
        nll = np.array([r["nll"] for _, r in recs]).sum(0); H = np.array([r["H"] for _, r in recs]).sum(0)
        V = np.array([r["V"] for _, r in recs]).sum(0); n = sum(r["n"] for _, r in recs)
        i = int(np.argmin(nll)); T = TEMPS[i]
        if 0 < i < len(TEMPS) - 1:
            c = np.polyfit(TEMPS[i - 1:i + 2], nll[i - 1:i + 2], 2)
            if c[0] > 0:
                T = float(-c[1] / (2 * c[0]))
        lp = np.concatenate([np.array(r["lp1"]) for _, r in recs if "lp1" in r])
        idr = [(k, r) for k, r in recs if k[1] in ("v1_ident", "v2_d_idv")]
        x = {"n_tok": n, "T": T, "exc07": (nll[I7] - H[I7]) / n * 1000, "z07": (nll[I7] - H[I7]) / math.sqrt(V[I7]),
             "exc10": (nll[6] - H[6]) / n * 1000,
             "tail002": float(np.mean(lp < math.log(0.002))) * 100, "tail05": float(np.mean(lp < math.log(0.05))) * 100,
             "exc_ident": (sum(r["nll"][I7] - r["H"][I7] for _, r in idr) / max(sum(r["n"] for _, r in idr), 1)) if idr else 0.0}
        x["tail_ratio"] = x["tail002"] / max(x["tail05"], 1e-6)
        x["llr_fp8"], x["z_fp8"] = _llr(d8, recs)
        x["llr_int8"], x["z_int8"] = _llr(di, recs)
        x["llr_prox"], x["z_prox"] = _llr(dp, recs)
        for v, dv in (("bf16", None), ("fp8w", d8), ("int8w", di)):
            x[f"rut_u_{v}"], x[f"rut_cvm_{v}"], x[f"rut_lp_{v}"] = _rut(recs, dv, pools[v], rng)
        X[key] = x
    return X
