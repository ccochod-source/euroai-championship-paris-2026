"""Étiqueteur à règles à partir des traits de vraisemblance (features_lik.py), et score simulé.

Règles, par famille :
  1. T* >= 0.95                      -> S (déguisement « sampling T = 1.0 »)
  2. z brut fp8 (fp8w ou fp8wa) > Z8 -> modèle fp8 (tous les endpoints fp8 de la famille = UN modèle)
  3. z brut int8w > ZI              -> modèle int8
  4. endpoint signalé persona/system prompt par le TEXTE (liste fournie) -> S
  5. sinon excès à T = 0.7 (x1e-3 nats/token) <= THR -> S, au-dessus -> LoRA ; chaque LoRA = un modèle distinct
     (forces différentes), sauf si deux excès sont à moins de MERGE l'un de l'autre.

    python sim/lik_labeler.py --features sim/out/lik_features_fake.csv --fake      (score avec la vérité du faux jeu)
    python sim/lik_labeler.py --features sim/out/lik_features_real.csv --persona ep-02,ep-08,ep-20,ep-16,ep-23
"""
import argparse
import itertools
import math

import numpy as np
import pandas as pd

REF_TRUTH = {"ref-01": "S", "ref-02": "T1", "ref-03": "S", "ref-04": "T3", "ref-05": "S", "ref-06": "S",
             "ref-07": "T4", "ref-08": "S", "ref-09": "T2"}


def label_family(df, persona, Z8=0.0, ZI=2.5, THR=90.0, MERGE=0.0, use_int8=True):
    lab = {}
    fam = df["family"].iloc[0]
    for _, r in df.iterrows():
        e = r["endpoint"]
        z8 = max(r.get("zraw_fp8w", -9), r.get("zraw_fp8wa", -9))
        if r["T_star"] >= 0.95:
            lab[e] = f"{fam}_S"
        elif z8 > Z8:
            lab[e] = f"{fam}_fp8"
        elif use_int8 and r.get("zraw_int8w", -9) > ZI:
            lab[e] = f"{fam}_int8"
        elif e in persona:
            lab[e] = f"{fam}_S"
        elif r["excess07"] <= THR:
            lab[e] = f"{fam}_S"
    rest = df[~df["endpoint"].isin(lab)].sort_values("excess07")
    k, prev = 0, None
    for _, r in rest.iterrows():
        if prev is None or r["excess07"] - prev > MERGE:
            k += 1
        lab[r["endpoint"]] = f"{fam}_L{k}"
        prev = r["excess07"]
    return lab


def mcc(t, g):
    t, g = np.asarray(t, bool), np.asarray(g, bool)
    tp = (t & g).sum(); tn = (~t & ~g).sum(); fp = (~t & g).sum(); fn = (t & ~g).sum()
    d = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def score(lab, truth, fam):
    eps = sorted(lab)
    tt, gg = [], []
    for a, b in itertools.combinations(eps, 2):
        if fam[a] != fam[b]:
            continue
        tt.append(truth[a] == truth[b]); gg.append(lab[a] == lab[b])
    return mcc(tt, gg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--persona", default="")
    ap.add_argument("--thr", type=float, default=90.0)
    ap.add_argument("--z8", type=float, default=0.0)
    a = ap.parse_args()
    df = pd.read_csv(a.features)
    if a.fake:
        for block in sorted(df["block"].unique()):
            d = df[df["block"] == block]
            persona = set(d[d["variant_true"].isin(["sysprompt", "persona"])]["endpoint"])
            truth = dict(zip(d["endpoint"], d["model_true"]))
            fam = dict(zip(d["endpoint"], d["family"]))
            lab = {}
            for f, g in d.groupby("family"):
                lab.update(label_family(g, persona, Z8=a.z8, THR=a.thr))
            for f, g in d.groupby("family"):
                sub = {e: lab[e] for e in g["endpoint"]}
                m = score(sub, truth, fam)
                print(f"bloc {block} famille {f}: MCC modèle = {m:.3f}")
                for _, r in g.sort_values("endpoint").iterrows():
                    print(f"   {r['endpoint']} vrai={r['model_true']:10s} ({r['variant_true']:12s}) -> {lab[r['endpoint']]:10s} "
                          f"T*={r['T_star']} exc={r['excess07']:+.0f} z8={max(r['zraw_fp8w'], r['zraw_fp8wa']):+.1f} zI={r['zraw_int8w']:+.1f}")
            m = score(lab, truth, fam)
            print(f"==> bloc {block} : MCC modèle global = {m:.3f} -> score simulé = {0.1 + 0.9 * m:.3f}")
    else:
        persona = set(x for x in a.persona.split(",") if x)
        for f, g in df.groupby("family"):
            lab = label_family(g, persona, Z8=a.z8, THR=a.thr)
            print(f"\n{f}:")
            for _, r in g.sort_values("endpoint").iterrows():
                print(f"   {r['endpoint']} -> {lab[r['endpoint']]:10s} T*={r['T_star']} exc={r['excess07']:+.0f} "
                      f"z8={max(r['zraw_fp8w'], r['zraw_fp8wa']):+.1f} zI={r['zraw_int8w']:+.1f}"
                      + ("  [persona/sys texte]" if r['endpoint'] in persona else ""))
            if f == "refs":
                fam = {e: "P" for e in lab}
                print(f"   MCC refs = {score(lab, REF_TRUTH, fam):.3f}")


if __name__ == "__main__":
    main()
