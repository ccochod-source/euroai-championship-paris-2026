"""Échantillonneur le plus vraisemblable par endpoint (sortie de lik.py --samplers).
Pour chaque configuration (T, top-k, top-p) : NLL pénalisée = NLL + PEN x (tokens hors support). On donne la meilleure
configuration, la meilleure sans troncature, le gain de la troncature (nats), et l'excès (NLL - H) sous la meilleure.
    python sim/analyze_samp.py [dir] [--fake spec.json]"""
import sys, glob, json, os, math, collections
import numpy as np
PEN = 12.0
d = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "sim/out/lik_samp"
truth = {}
if "--fake" in sys.argv:
    truth = {e["ep"]: e["variant"] for e in json.load(open(sys.argv[sys.argv.index("--fake") + 1]))["endpoints"]}
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(4)))
    ntok = collections.Counter()
    for l in open(p):
        r = json.loads(l)
        if "samp" not in r:
            continue
        ntok[r["endpoint"]] += r["n"]
        for k, v in r["samp"].items():
            acc[r["endpoint"]][k] += np.array(v)
    if not acc:
        continue
    print(f"\n### {os.path.basename(p)[:-6]}")
    print(f"{'endpoint':8s} {'vrai':12s} {'meilleur (T|k|p)':>17s} {'hors sup.':>9s} {'excès':>6s} {'z':>6s} | {'meilleur sans tronc.':>20s} {'excès':>6s} {'z':>6s} | {'gain tronc. (nats)':>18s} | hors support à T=0.7 : k50 / p0.95 / p0.9")
    for e in sorted(acc):
        A = acc[e]; n = ntok[e]
        pen = {k: v[0] + PEN * v[3] for k, v in A.items()}
        best = min(pen, key=pen.get)
        nt = {k: v for k, v in pen.items() if k.endswith("|0|1.0")}
        bnt = min(nt, key=nt.get)
        def ex(k):
            v = A[k]; return (v[0] - v[1]) / n * 1000, (v[0] - v[1]) / math.sqrt(max(v[2], 1e-9))
        e1, z1 = ex(best); e2, z2 = ex(bnt)
        print(f"{e:8s} {truth.get(e, ''):12s} {best:>17s} {int(A[best][3]):9d} {e1:+6.0f} {z1:+6.1f} | {bnt:>20s} {e2:+6.0f} {z2:+6.1f} | {pen[bnt] - pen[best]:18.1f} | "
              f"{int(A['0.7|50|1.0'][3])} / {int(A['0.7|0|0.95'][3])} / {int(A['0.7|0|0.9'][3])}   (n={n})")
