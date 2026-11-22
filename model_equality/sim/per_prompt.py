"""Excès de surprise (NLL - H à T = 0.7, x1e-3 nats/token) par endpoint et par groupe de sondes, centré sur la médiane
de la famille pour ce groupe. Montre SUR QUELLES sondes un endpoint s'écarte (domaine d'un LoRA, system prompt...).
    python sim/per_prompt.py [dir]"""
import sys, re, collections
import numpy as np
sys.path.insert(0, "sim")
import analyze_lik as A
from features_lik import REAL
d = sys.argv[1] if len(sys.argv) > 1 else "sim/out/lik_r2"
data = A.load(d)
i7 = A.TEMPS.index(0.7)
def grp(pid):
    pid = re.sub(r"\d+$", "", pid)
    for p in ("v2_c_fr", "v2_pk_mul", "v2_pk_mix", "v2_cat", "v1_rand", "v1_fr", "v2b_story", "v2b_pk_mul", "v2b_x", "v2b_tr", "v2b_cont"):
        if pid.startswith(p): return p
    return pid
for fam, (m, eps) in REAL.items():
    rs = "refs" if fam == "refs" else "eps"
    eps = [e if e.startswith("ref") else f"ep-{e}" for e in eps.split()]
    db = data.get((m, "bf16", rs))
    if not db: continue
    acc = collections.defaultdict(lambda: [0.0, 0])
    for (e, pid, rep), r in db.items():
        a = acc[(e, grp(pid))]; a[0] += r["nll"][i7] - r["H"][i7]; a[1] += r["n"]
    groups = sorted({g for (_, g) in acc})
    groups = [g for g in groups if sum((e, g) in acc for e in eps) == len(eps)]
    M = np.array([[acc[(e, g)][0] / max(acc[(e, g)][1], 1) * 1000 for g in groups] for e in eps])
    N = np.array([[acc[(e, g)][1] for g in groups] for e in eps])
    C = M - np.median(M, 0)
    print(f"\n### {fam} : excès par groupe de sondes, centré sur la médiane de la famille (x1e-3 nats/token) ; médiane brute en dernière ligne")
    print("endpoint  " + " ".join(f"{g[-9:]:>9s}" for g in groups))
    for e, row in zip(eps, C):
        print(f"{e:9s} " + " ".join(f"{v:+9.0f}" for v in row))
    print("médiane   " + " ".join(f"{v:+9.0f}" for v in np.median(M, 0)))
    print("tokens    " + " ".join(f"{v:9.0f}" for v in np.median(N, 0)))
