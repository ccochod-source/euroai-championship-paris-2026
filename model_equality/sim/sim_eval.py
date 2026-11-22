"""Évalue les méthodes de l'équipe (src/methods/*.py ou sim/methods/*.py) sur un FAUX jeu à vérité connue.

Hypothèse : familles parfaites (identité = facile). Le score simulé est donc 0.1 + 0.9 * MCC_modèle, MCC calculé sur
les paires d'une même famille, après clustering average-linkage par famille.
Seuils : (a) oracle global (meilleur seuil unique sur le faux jeu) ; (b) calibré sur refsim (règle d'evaluate.py :
milieu entre max D(même modèle) et min D(différent)) puis appliqué tel quel au faux jeu (transfert refs -> eps).
Chaque faux endpoint a 2 blocs indépendants de réponses (repeat < 1000 et >= 1000) : on évalue les deux.

    python sim/sim_eval.py [--set sim/out/fake/set1] [--refsim sim/out/fake/refsim] [methods...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import itertools
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
REF_TRUTH = {"ref-01": "S", "ref-02": "T1", "ref-03": "S", "ref-04": "T3", "ref-05": "S", "ref-06": "S",
             "ref-07": "T4", "ref-08": "S", "ref-09": "T2"}


def load_method(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem + "_m", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


def load_fake(set_dir, block, select=None):
    data = {}
    for p in sorted(glob.glob(os.path.join(set_dir, "raw", "*.jsonl"))):
        ep = os.path.basename(p)[:-6]
        recs = []
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["repeat"] // 1000 != block:
                continue
            r = dict(r, repeat=r["repeat"] % 1000)
            if select is None or select(r):
                recs.append(r)
        data[ep] = recs
    return data


def mcc(t, g):
    t, g = np.asarray(t, bool), np.asarray(g, bool)
    tp = (t & g).sum(); tn = (~t & ~g).sum(); fp = (~t & g).sum(); fn = (t & ~g).sum()
    d = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def clean(D):
    D = np.array(D, float)
    if np.isnan(D).any():
        D[np.isnan(D)] = np.nanmean(D)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    return np.maximum(D, 0)


def labels_within(D, eps, fam, t):
    lab = {}
    for f in sorted(set(fam.values())):
        idx = [i for i, e in enumerate(eps) if fam[e] == f]
        if len(idx) == 1:
            lab[eps[idx[0]]] = f"{f}_0"
            continue
        sub = D[np.ix_(idx, idx)]
        cl = fcluster(linkage(squareform(sub, checks=False), "average"), t, "distance")
        for i, c in zip(idx, cl):
            lab[eps[i]] = f"{f}_{c}"
    return lab


def model_mcc(lab, eps, fam, model):
    tt, gg = [], []
    for a, b in itertools.combinations(eps, 2):
        if fam[a] != fam[b]:
            continue
        tt.append(model[a] == model[b]); gg.append(lab[a] == lab[b])
    return mcc(tt, gg)


def auc_within(D, eps, fam, model):
    pos, neg = [], []
    for i, j in itertools.combinations(range(len(eps)), 2):
        a, b = eps[i], eps[j]
        if fam[a] != fam[b]:
            continue
        (pos if model[a] == model[b] else neg).append(D[i, j])
    pos, neg = np.array(pos), np.array(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def oracle(D, eps, fam, model):
    cands = np.unique(D[np.triu_indices(len(eps), 1)])
    ts = np.concatenate([[0], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1]]) if len(cands) > 1 else [0]
    best = (0.0, 0.0)
    for t in ts:
        m = model_mcc(labels_within(D, eps, fam, t), eps, fam, model)
        if m > best[0]:
            best = (m, t)
    return best


def calib_t(D, eps, truth):
    same = [D[i, j] for i, j in itertools.combinations(range(len(eps)), 2) if truth[eps[i]] == truth[eps[j]]]
    diff = [D[i, j] for i, j in itertools.combinations(range(len(eps)), 2) if truth[eps[i]] != truth[eps[j]]]
    return (max(same) + min(diff)) / 2


def evaluate(mod, set_dir, refsim_dir, truth_csv):
    t = {r["endpoint"]: r for r in csv.DictReader(open(truth_csv))}
    fams_ok = os.environ.get("SIM_FAMS")  # ex. "LL1,QB15" : restreint aux familles complètes
    if fams_ok:
        t = {e: r for e, r in t.items() if r["family"] in fams_ok.split(",")}
    eps = sorted(t)
    fam = {e: t[e]["family"] for e in eps}
    model = {e: t[e]["model"] for e in eps}
    out = []
    for block in (0, 1):
        data = load_fake(set_dir, block, mod.select)
        data = {e: data.get(e, []) for e in eps}
        if sum(len(v) for v in data.values()) == 0:
            return None
        D = clean(mod.distance_matrix(data, eps))
        auc = auc_within(D, eps, fam, model)
        om, ot = oracle(D, eps, fam, model)
        cm = float("nan")
        if refsim_dir and os.path.isdir(os.path.join(refsim_dir, "raw")):
            rdata = load_fake(refsim_dir, block, mod.select)
            reps = sorted(rdata)
            try:
                Dr = clean(mod.distance_matrix(rdata, reps))
                tc = calib_t(Dr, reps, REF_TRUTH)
                cm = model_mcc(labels_within(D, eps, fam, tc), eps, fam, model)
            except Exception as e:  # noqa
                cm = float("nan")
        n_req = np.mean([len(v) for v in data.values()])
        out.append(dict(auc=auc, oracle_mcc=om, oracle_t=ot, calib_mcc=cm, n_req=n_req, D=D))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(ROOT / "sim/out/fake/set1"))
    ap.add_argument("--refsim", default=str(ROOT / "sim/out/fake/refsim"))
    ap.add_argument("--truth", default=str(ROOT / "sim/truth_set1.csv"))
    ap.add_argument("methods", nargs="*")
    a = ap.parse_args()
    paths = a.methods or sorted(glob.glob(str(ROOT / "src/methods/*.py"))) + sorted(glob.glob(str(ROOT / "sim/methods/*.py")))
    print("| méthode | req/ep | AUC intra-famille (b0/b1) | MCC modèle oracle (b0/b1) | score oracle | MCC seuil refsim (b0/b1) | score refsim |")
    print("|---|---|---|---|---|---|---|")
    for p in paths:
        try:
            mod = load_method(p)
            r = evaluate(mod, a.set, a.refsim, a.truth)
        except Exception as e:
            print(f"| {Path(p).stem} | ERREUR {type(e).__name__}: {str(e)[:80]} |")
            continue
        if r is None:
            print(f"| {Path(p).stem} | pas de données |")
            continue
        f = lambda k: "/".join(f"{x[k]:.2f}" for x in r)
        so = np.mean([0.1 + 0.9 * x["oracle_mcc"] for x in r])
        sc = np.nanmean([0.1 + 0.9 * x["calib_mcc"] for x in r])
        print(f"| {Path(p).stem} | {r[0]['n_req']:.0f} | {f('auc')} | {f('oracle_mcc')} | {so:.3f} | {f('calib_mcc')} | {sc:.3f} |",
              flush=True)


if __name__ == "__main__":
    main()
