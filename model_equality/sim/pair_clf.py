"""Classifieur de paires « même modèle / différent », entraîné sur le jeu n° 3 (vrais checkpoints, vérité connue),
appliqué aux vrais endpoints. Traits par instance : sim/features3.py ; traits de paire : |a-b|, min, max.

    python sim/pair_clf.py --budget v1,v2a [--real-dir sim/out/lik_r3] [--out results/sim_pair_probs.csv]
Validation : (1) une famille laissée de côté à tour de rôle ; (2) vrais refs (SmolLM, vérité connue) quand S360 est prêt.
"""
import argparse, collections, itertools, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features3 as F3
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef

REAL = {"LL1": "05 09 10 11 13 15 16 19 27 29 31", "QA3": "01 02 04 12 14 17 18 23 24 25 33",
        "QB15": "03 06 07 08 20 21 22 26 28 30 32"}
REF_TRUTH = {"ref-01": "S", "ref-02": "T1", "ref-03": "S", "ref-04": "T3", "ref-05": "S", "ref-06": "S",
             "ref-07": "T4", "ref-08": "S", "ref-09": "T2"}
FEATS = ["T", "exc07", "z07", "exc10", "tail002", "tail05", "tail_ratio", "exc_ident", "llr_fp8", "z_fp8", "llr_int8", "z_int8",
         "llr_prox", "z_prox", "rut_u_bf16", "rut_cvm_bf16", "rut_lp_bf16", "rut_u_fp8w", "rut_cvm_fp8w", "rut_u_int8w", "rut_cvm_int8w"]


def pair_vec(a, b):
    v = []
    for f in FEATS:
        x, y = a.get(f, np.nan), b.get(f, np.nan)
        v += [abs(x - y), min(x, y), max(x, y)]
    return v


def sim_pairs(X, truth, rng, max_pairs=40000):
    keys = sorted(X)
    P, Y = [], []
    pairs = [(a, b) for a, b in itertools.combinations(keys, 2) if not (a[0] == b[0] and a[1] == b[1])]
    rng.shuffle(pairs)
    for a, b in pairs[:max_pairs]:
        P.append(pair_vec(X[a], X[b])); Y.append(int(truth[a[0]] == truth[b[0]]))
    return np.array(P), np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="v1,v2a")
    ap.add_argument("--lik3", default="sim/out/lik3")
    ap.add_argument("--real-dir", default="sim/out/lik_r3")
    ap.add_argument("--spec", default="sim/spec_set3.json")
    ap.add_argument("--out", default="results/sim_pair_probs.csv")
    a = ap.parse_args()
    srcs = tuple(a.budget.split(","))
    spec = json.load(open(a.spec))["endpoints"]
    truth = {e["ep"]: e["model"] for e in spec}
    rng = np.random.default_rng(0)
    data = {}
    for fam in F3.BASE:
        X = F3.build(a.lik3, a.lik3, fam, "all_inst", srcs)
        if X:
            data[fam] = X
            print(f"{fam}: {len(X)} instances simulées")
    fams = sorted(data)
    if len(fams) < 2:
        print("pas assez de familles simulées"); return
    sets = {f: sim_pairs(data[f], truth, rng) for f in fams}
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced")
    print("\n## Validation croisée par famille (entraînement sur les autres familles)")
    for f in fams:
        Xtr = np.vstack([sets[g][0] for g in fams if g != f]); Ytr = np.concatenate([sets[g][1] for g in fams if g != f])
        clf = mk().fit(Xtr, Ytr)
        p = clf.predict_proba(sets[f][0])[:, 1]
        print(f"  {f}: AUC {roc_auc_score(sets[f][1], p):.3f}  MCC(0.5) {matthews_corrcoef(sets[f][1], p > 0.5):.3f}  ({len(p)} paires, {sets[f][1].mean():.2f} positives)")
    clf = mk().fit(np.vstack([sets[g][0] for g in fams]), np.concatenate([sets[g][1] for g in fams]))
    rows = []
    # vrais refs
    R = F3.build(a.real_dir, a.lik3, "S360", "refs", srcs, blocks=False)
    if R:
        eps = sorted(k[0] for k in R)
        t, pr = [], []
        for x, y in itertools.combinations(eps, 2):
            p = clf.predict_proba([pair_vec(R[(x, 0)], R[(y, 0)])])[0, 1]
            t.append(REF_TRUTH[x] == REF_TRUTH[y]); pr.append(p); rows.append(("refs", x, y, p))
        print(f"\n## Vrais refs : AUC {roc_auc_score(t, pr):.3f}  MCC(0.5) {matthews_corrcoef(t, np.array(pr) > 0.5):.3f}")
    for fam, s in REAL.items():
        eps = [f"ep-{x}" for x in s.split()]
        R = F3.build(a.real_dir, a.lik3, fam, "eps", srcs, eps=set(eps), blocks=False)
        if not R:
            continue
        eps = sorted(k[0] for k in R)
        print(f"\n## {fam} : P(même modèle) par paire")
        M = np.eye(len(eps))
        for i, j in itertools.combinations(range(len(eps)), 2):
            p = clf.predict_proba([pair_vec(R[(eps[i], 0)], R[(eps[j], 0)])])[0, 1]
            M[i, j] = M[j, i] = p; rows.append((fam, eps[i], eps[j], p))
        print("       " + " ".join(e[-2:].rjust(4) for e in eps))
        for i, e in enumerate(eps):
            print(f"{e}  " + " ".join(f"{M[i, j]:4.2f}" for j in range(len(eps))))
    if rows:
        with open(a.out, "w") as f:
            f.write("family,a,b,p_same\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f}\n")
        print("->", a.out)


if __name__ == "__main__":
    main()
