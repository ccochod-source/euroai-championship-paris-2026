"""Fidélité du simulateur : métriques des méthodes de l'équipe sur les VRAIS refs (data/raw) vs les refs SIMULÉS (refsim).

    python sim/sim_eval_refs.py [methods...]
"""
import glob, itertools, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_eval import load_method, load_fake, clean, mcc, REF_TRUTH, ROOT
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
REFS = sorted(REF_TRUTH)
PAIRS = list(itertools.combinations(range(9), 2))
SAME = np.array([REF_TRUTH[REFS[i]] == REF_TRUTH[REFS[j]] for i, j in PAIRS])

def load_real(select):
    data = {}
    for e in REFS:
        data[e] = [r for r in (json.loads(l) for l in open(ROOT / "data/raw" / f"{e}.jsonl", encoding="utf-8")) if select(r)]
    return data

def metrics(D):
    d = np.array([D[i, j] for i, j in PAIRS])
    pos, neg = d[SAME], d[~SAME]
    auc = ((pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    Z = linkage(squareform(D, checks=False), "average")
    best = 0
    for t in np.unique(d):
        for tt in (t - 1e-9, t + 1e-9):
            lab = fcluster(Z, tt, "distance")
            best = max(best, mcc(SAME, [lab[i] == lab[j] for i, j in PAIRS]))
    i2, i9 = REFS.index("ref-02"), REFS.index("ref-09")
    S = [REFS.index(e) for e in REFS if REF_TRUTH[e] == "S"]
    maxS = max(D[i, j] for i in S for j in S if i < j)
    sep = lambda x: min(D[x, s] for s in S) / maxS if maxS > 0 else float("nan")
    return auc, best, sep(i2), sep(i9)

paths = sys.argv[1:] or sorted(glob.glob(str(ROOT / "src/methods/*.py"))) + sorted(glob.glob(str(ROOT / "sim/methods/*.py")))
print("| méthode | n réel | AUC réel | AUC sim b0/b1 | MCC oracle réel | MCC sim b0/b1 | sep fp8 réel / sim | sep L0.3 réel / sim |")
print("|---|---|---|---|---|---|---|---|")
for p in paths:
    try:
        m = load_method(p)
        real = load_real(m.select)
        nreal = np.mean([len(v) for v in real.values()])
        rr = metrics(clean(m.distance_matrix(real, REFS))) if min(len(v) for v in real.values()) > 0 else None
        ss = [metrics(clean(m.distance_matrix(load_fake(str(ROOT / "sim/out/fake/refsim"), b, m.select), REFS))) for b in (0, 1)]
    except Exception as e:
        print(f"| {Path(p).stem} | ERREUR {type(e).__name__}: {str(e)[:70]} |"); continue
    f = lambda x, k: f"{x[k]:.2f}" if x else "—"
    print(f"| {Path(p).stem} | {nreal:.0f} | {f(rr,0)} | {ss[0][0]:.2f}/{ss[1][0]:.2f} | {f(rr,1)} | {ss[0][1]:.2f}/{ss[1][1]:.2f} | "
          f"{f(rr,2)} / {ss[0][2]:.2f} | {f(rr,3)} / {ss[0][3]:.2f} |", flush=True)
