"""Exporte les traits de vraisemblance par endpoint (réels ou faux) dans un CSV, pour l'orchestrateur.

    python sim/features_lik.py --dir sim/out/lik --out sim/out/lik_features_real.csv
    python sim/features_lik.py --dir sim/out/lik_fake --fake sim/spec_set1.json --out sim/out/lik_features_fake.csv

Colonnes : endpoint, famille, checkpoint, n_tok, T_star (MLE), nll_tok (à T*), excess07 (NLL-H à T=0.7, x1e-3/token),
z07 (z d'adéquation à T=0.7), puis pour chaque variante v : llr_v (x1e-3/token à T=0.7, centré sur la médiane de la
famille) et zraw_v (z brut, non centré).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_lik as A  # noqa: E402

REAL = {
    "Llama": ("unsloth/Llama-3.2-1B-Instruct", "05 09 10 11 13 15 16 19 27 29 31"),
    "QwenA": ("Qwen/Qwen2.5-3B-Instruct", "01 02 04 12 14 17 18 23 24 25 33"),
    "QwenB": ("Qwen/Qwen2.5-1.5B-Instruct", "03 06 07 08 20 21 22 26 28 30 32"),
    "refs": ("HuggingFaceTB/SmolLM2-360M-Instruct", " ".join(f"ref-0{i}" for i in range(1, 10))),
}
VARS = ["fp8w", "fp8wa", "fp8ch", "int8w", "bnb8", "nf4", "int4g128"]


def fam_table(data, model, eps, recset, block=None, T=0.7):
    key = (model, "bf16", recset)
    if key not in data:
        return []
    filt = (lambda k: True) if block is None else (lambda k: int(k[2]) // 1000 == block)
    db = {k: r for k, r in data[key].items() if filt(k)}
    i7 = A.TEMPS.index(T)
    rows = []
    for e in eps:
        g = [r for k, r in db.items() if k[0] == e]
        if not g:
            continue
        Ts, nl, z, n = A.fit_T(g)
        nll = sum(r["nll"][i7] for r in g); H = sum(r["H"][i7] for r in g); V = sum(r["V"][i7] for r in g)
        rows.append({"endpoint": e, "checkpoint": model.split("/")[-1], "n_tok": n, "T_star": Ts, "nll_tok": round(nl, 4),
                     "excess07": round((nll - H) / n * 1000, 1), "z07": round((nll - H) / np.sqrt(V), 2)})
    for v in VARS:
        if (model, v, recset) not in data:
            continue
        dq = {k: r for k, r in data[(model, v, recset)].items() if filt(k)}
        vals = {r["endpoint"]: A.llr(dq, db, r["endpoint"], T) for r in rows}
        med = np.nanmedian([m for m, _ in vals.values()])
        for r in rows:
            m, z = vals[r["endpoint"]]
            r[f"llr_{v}"] = round((m - med) * 1000, 1)
            r[f"zraw_{v}"] = round(z, 2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sim/out/lik")
    ap.add_argument("--fake", default=None, help="spec JSON du faux jeu (sinon : vrais endpoints)")
    ap.add_argument("--refsim", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = A.load(a.dir)
    rows = []
    if a.fake:
        spec = json.load(open(a.fake))
        fams = {}
        for e in spec["endpoints"]:
            fams.setdefault(e["family"], (e["base"], []))[1].append(e)
        truth = {e["ep"]: e for e in spec["endpoints"]}
        for f, (base, es) in fams.items():
            recset = "refs" if es[0]["ep"].startswith("ref") else "eps"
            for b in (0, 1):
                for r in fam_table(data, base, sorted(x["ep"] for x in es), recset, block=b):
                    t = truth[r["endpoint"]]
                    rows.append({"family": f, "block": b, "model_true": t["model"], "variant_true": t.get("variant", t["weights"]), **r})
    else:
        for f, (m, eps) in REAL.items():
            eps = [e if e.startswith("ref") else f"ep-{e}" for e in eps.split()]
            for r in fam_table(data, m, eps, "refs" if f == "refs" else "eps"):
                rows.append({"family": f, **r})
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} lignes -> {a.out}")


if __name__ == "__main__":
    main()
