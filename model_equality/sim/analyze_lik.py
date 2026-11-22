"""Analyse des log-vraisemblances (sortie de lik.py) : identification du checkpoint, température, tests fp8/int.

    python sim/analyze_lik.py [--dir sim/out/lik] [--fam real|set1] [--truth sim/truth_set1.csv]
"""
import argparse
import collections
import glob
import json
import math
import os

import numpy as np

TEMPS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
REAL_FAM = {
    "Llama": "05 09 10 11 13 15 16 19 27 29 31",
    "QwenA": "01 02 04 06 12 14 17 18 23 24 25 33",
    "QwenB": "03 07 08 20 21 22 26 28 30 32",
}
REF_TRUTH = {"ref-01": "S plain", "ref-02": "T1 fp8", "ref-03": "S sys", "ref-04": "T3 L0.7", "ref-05": "S persona",
             "ref-06": "S wm", "ref-07": "T4 L1.5", "ref-08": "S T1.0", "ref-09": "T2 L0.3"}


def load(d):
    """{(model, variant, records): {key: rec}}"""
    out = {}
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        b = os.path.basename(p)[:-6]
        model, variant, recs = b.split("__")[0] + "/" + b.split("__")[1], b.split("__")[2], b.split("__")[3]
        dd = {}
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            dd[(r["endpoint"], r["prompt_id"], r["repeat"])] = r
        out[(model, variant, recs)] = dd
    return out


def per_ep(dd, keyfilter=None):
    g = collections.defaultdict(list)
    for k, r in dd.items():
        if keyfilter is None or keyfilter(k):
            g[k[0]].append(r)
    return g


def fit_T(recs):
    nll = np.array([r["nll"] for r in recs]).sum(0)
    H = np.array([r["H"] for r in recs]).sum(0)
    V = np.array([r["V"] for r in recs]).sum(0)
    n = sum(r["n"] for r in recs)
    i = int(np.argmin(nll))
    z = (nll[i] - H[i]) / math.sqrt(max(V[i], 1e-9))
    return TEMPS[i], nll[i] / n, z, n


LLR_T = None  # None : log-probs T=1 ; sinon température de la grille TEMPS (ex. 0.7)


def llr(dq, db, ep, T=None):
    """Moyenne par token de log p_q - log p_bf16 (à T=1, ou à la température T) et z (SE sur les enregistrements)."""
    T = LLR_T if T is None else T
    vals, ns = [], []
    for k, rb in db.items():
        if k[0] != ep or k not in dq:
            continue
        if T is None:
            a, b = np.array(dq[k]["lp1"]), np.array(rb["lp1"])
            if len(a) != len(b) or len(a) == 0:
                continue
            vals.append((a - b).sum())
            ns.append(len(a))
        else:
            i = TEMPS.index(T)
            if dq[k]["n"] != rb["n"] or rb["n"] == 0:
                continue
            vals.append(rb["nll"][i] - dq[k]["nll"][i])
            ns.append(rb["n"])
    if not vals:
        return float("nan"), float("nan")
    vals, ns = np.array(vals), np.array(ns)
    m = vals.sum() / ns.sum()
    # SE d'un ratio : résidus par enregistrement
    res = vals - m * ns
    se = math.sqrt((res ** 2).sum()) / ns.sum() if len(vals) > 1 else float("nan")
    return m, m / se if se and se == se else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "out", "lik"))
    ap.add_argument("--fam", default="real")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    data = load(a.dir)
    models = sorted({m for m, v, r in data})
    lines = []
    P = lambda s="": lines.append(s)

    if a.fam == "real":
        fam_of = {f"ep-{x}": f for f, s in REAL_FAM.items() for x in s.split()}
    else:
        import csv
        t = {r["endpoint"]: r for r in csv.DictReader(open(a.truth))}
        fam_of = {e: r["family"] for e, r in t.items()}
        label = {e: r["variant"] for e, r in t.items()}

    # ---------- 1. identification du checkpoint (bf16, meilleure T) ----------
    P("## 1. Identification du checkpoint : NLL/token à la meilleure température (bf16)")
    for recset in ("refs", "eps"):
        cands = [m for m in models if (m, "bf16", recset) in data]
        if not cands:
            continue
        eps = sorted({k[0] for m in cands for k in data[(m, "bf16", recset)]})
        P(f"\n### {recset}\n")
        P("| endpoint | groupe | " + " | ".join(c.split("/")[-1] for c in cands) + " | meilleur |")
        P("|---|---|" + "---|" * (len(cands) + 1))
        for ep in eps:
            row = []
            for c in cands:
                g = per_ep(data[(c, "bf16", recset)])[ep]
                row.append(fit_T(g)[1] if g else float("nan"))
            best = cands[int(np.nanargmin(row))].split("/")[-1]
            grp = REF_TRUTH.get(ep, fam_of.get(ep, "?"))
            P(f"| {ep} | {grp} | " + " | ".join(f"{x:.3f}" for x in row) + f" | {best} |")

    # ---------- 2. température, adéquation, diagnostics d'échantillonnage ----------
    P("\n## 2. Température MLE, z d'adéquation, diagnostics top-k/top-p (sous le meilleur candidat bf16)")
    for recset in ("refs", "eps"):
        cands = [m for m in models if (m, "bf16", recset) in data]
        if not cands:
            continue
        eps = sorted({k[0] for m in cands for k in data[(m, "bf16", recset)]})
        P(f"\n### {recset}\n")
        P("| endpoint | groupe | modèle | T* | NLL/tok | z adéq. | rang max | frac rang>=20 | masse-au-dessus max | n tok |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for ep in eps:
            best, bv = None, 1e9
            for c in cands:
                g = per_ep(data[(c, "bf16", recset)])[ep]
                if g:
                    v = fit_T(g)[1]
                    if v < bv:
                        best, bv = c, v
            g = per_ep(data[(best, "bf16", recset)])[ep]
            T, nl, z, n = fit_T(g)
            mr = max(r["max_rank"] for r in g)
            fr = sum(r.get("n_rank_gt50", 0) for r in g) / max(n, 1)
            ma = max(r["max_above"] for r in g)
            grp = REF_TRUTH.get(ep, fam_of.get(ep, "?"))
            if a.fam != "real":
                grp += " " + label.get(ep, "")
            P(f"| {ep} | {grp} | {best.split('/')[-1]} | {T} | {nl:.3f} | {z:+.1f} | {mr} | {fr:.4f} | {ma:.3f} | {n} |")

    # ---------- 3. tests de rapport de vraisemblance quantifié vs bf16 ----------
    P("\n## 3. LLR par token log p_variante - log p_bf16 (T=1), z entre parenthèses. Positif = la variante explique mieux.")
    for (m, v0, recset) in sorted(data):
        if v0 != "bf16":
            continue
        variants = sorted(v for (mm, v, rr) in data if mm == m and rr == recset and v != "bf16")
        if not variants:
            continue
        db = data[(m, "bf16", recset)]
        eps = sorted({k[0] for k in db})
        if recset == "eps":
            fams = {fam_of.get(e) for e in eps}
        P(f"\n### {m} ({recset})\n")
        P("| endpoint | groupe | " + " | ".join(variants) + " |")
        P("|---|---|" + "---|" * len(variants))
        for ep in eps:
            grp = REF_TRUTH.get(ep, fam_of.get(ep, "?"))
            if a.fam != "real":
                grp += " " + label.get(ep, "")
            cells = []
            for v in variants:
                mm, z = llr(data[(m, v, recset)], db, ep)
                cells.append(f"{mm * 1000:+.1f} ({z:+.1f})")
            P(f"| {ep} | {grp} | " + " | ".join(cells) + " |")
        P("\n(valeurs ×1000 nats/token)")
    out = "\n".join(lines)
    print(out)
    return out


if __name__ == "__main__":
    main()
