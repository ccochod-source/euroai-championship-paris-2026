"""Offline post-processing study on the saved train maps (no decoder query): t-rule, t-rule gated by the whole-image
margin (predict all-ones when the margin is high = tiny hole), and removal of small predicted hole components.
usage: post.py <method>"""
import sys
import cv2
from lib import *

meth = sys.argv[1]; set_log(f"post_{meth}")
z = np.load(f"{OUT}/occl_{meth}_train.npz"); names = [str(n) for n in z["names"]]; maps = z["maps"].astype(np.float32); m0 = z["m0"]
G = gt128(); gts = [G[n] for n in names]; N = len(names)
hole = np.array([1 - g.mean() for g in gts])
TS_ = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TAU = [np.inf] + list(np.quantile(m0, [0.5, 0.6, 0.7, 0.8, 0.9]))
AMIN = [0, 20, 50, 100, 200]


def clean(pred, amin):
    """Drop predicted hole components smaller than amin cells."""
    if amin == 0:
        return pred
    n, lab, st, _ = cv2.connectedComponentsWithStats((~pred).astype(np.uint8), connectivity=8)
    out = pred.copy()
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < amin:
            out[lab == k] = True
    return out


keys = [(t, tau, a) for t in TS_ for tau in TAU for a in AMIN]
S = np.zeros((N, len(keys)))
for i in range(N):
    base = {(t, a): miou(clean(maps[i] >= t, a), gts[i]) for t in TS_ for a in AMIN}
    one = miou(np.ones((128, 128)), gts[i])
    for j, (t, tau, a) in enumerate(keys):
        S[i, j] = one if m0[i] >= tau else base[(t, a)]
rng = np.random.RandomState(0); perm = rng.permutation(N); fold = np.empty(N, int); fold[perm] = np.arange(N) % 5


def cv(idx, lab):
    out = np.zeros(N); ch = []
    for f in range(5):
        j = idx[int(np.argmax(S[fold != f][:, idx].mean(0)))]; out[fold == f] = S[fold == f, j]; ch.append(keys[j])
    b = " ".join(f"{l}:{out[(hole >= a) & (hole < bb)].mean():.3f}" for a, bb, l in BINS)
    print(f"{meth} {lab:34s} cv mIoU {out.mean():.3f} | {b} | chosen {[(c[0], round(float(c[1]), 2), c[2]) for c in ch]}")


print(f"== post {meth}: all-ones {np.mean([miou(np.ones((128, 128)), g) for g in gts]):.3f}; corr(m0, hole) = {np.corrcoef(m0, hole)[0, 1]:.2f}")
cv([j for j, k in enumerate(keys) if k[1] == np.inf and k[2] == 0], "t-rule")
cv([j for j, k in enumerate(keys) if k[2] == 0], "t-rule + margin gate")
cv([j for j, k in enumerate(keys) if k[1] == np.inf], "t-rule + small-component removal")
cv(list(range(len(keys))), "t-rule + gate + removal")
