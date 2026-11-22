"""Score candidate maps from exp1_maps.py against the train masks (official per-image mIoU).
For each method x map: best global threshold (absolute), and per-image Otsu. Plus trivial baselines.
Usage: python eval_maps.py maps_train.npz
"""
import sys

import numpy as np


def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = np.logical_or(p == c, g == c).sum()
        s += 1.0 if u == 0 else np.logical_and(p == c, g == c).sum() / u
    return s / 2


def otsu(x):
    v = x[np.isfinite(x)].ravel()
    hist, edges = np.histogram(v, 64)
    w = hist.cumsum(); w2 = w[-1] - w
    mids = (edges[:-1] + edges[1:]) / 2
    m1 = (hist * mids).cumsum() / np.maximum(w, 1)
    m2 = ((hist * mids).sum() - (hist * mids).cumsum()) / np.maximum(w2, 1)
    k = np.argmax(w * w2 * (m1 - m2) ** 2)
    return mids[k]


d = np.load(sys.argv[1])
names, meth, gt = d["names"], d["method"], d["gt"]
keys = [k for k in d.files if k.startswith("map_")]
print("frac watermarked (gt) per method:")
for m in np.unique(meth):
    sel = meth == m
    print(f"  {m:11s} n={sel.sum()} mean_frac={gt[sel].mean():.3f}  empty={np.mean(gt[sel].sum((1,2))==0):.2f} "
          f"full={np.mean(gt[sel].mean((1,2))>0.99):.2f}")
print()
print(f"{'method':11s} {'map':8s} {'all0':>6s} {'all1':>6s} {'bestT':>6s} {'otsu':>6s} {'zT':>6s}")
for m in np.unique(meth):
    sel = np.where(meth == m)[0]
    g = gt[sel]
    a0 = np.mean([miou(np.zeros_like(x), x) for x in g]); a1 = np.mean([miou(np.ones_like(x), x) for x in g])
    for k in keys:
        M = d[k][sel]
        if np.isnan(M).all():
            continue
        qs = np.nanquantile(M, np.linspace(0.02, 0.98, 49))
        best = max((np.mean([miou((x > t).astype(np.uint8), y) for x, y in zip(M, g)]), t) for t in qs)
        ot = np.mean([miou((x > otsu(x)).astype(np.uint8), y) for x, y in zip(M, g)])
        # per-image z-score then global threshold
        Z = (M - M.mean((1, 2), keepdims=True)) / (M.std((1, 2), keepdims=True) + 1e-8)
        zb = max(np.mean([miou((x > t).astype(np.uint8), y) for x, y in zip(Z, g)]) for t in np.linspace(-1.5, 1.5, 31))
        print(f"{m:11s} {k[4:]:8s} {a0:6.3f} {a1:6.3f} {best[0]:6.3f} {ot:6.3f} {zb:6.3f}   (T={best[1]:.3f})")
