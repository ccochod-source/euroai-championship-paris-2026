"""CPU-only learned fusion of the weak evidence maps for whole-frame images (holes), per method.
Inputs per 4x4 cell (all 128 x 128): occlusion map (decoder), generator-comb log-likelihood L and its variance V
(session 87), decoder-feature head probability F (session 87), each also smoothed at a few scales.
Model: a small convolutional net (CPU), image-grouped 5-fold cross-validation on the real full train images.
usage: cellfuse.py <method> [epochs]"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from common import *
meth = sys.argv[1]; EP = int(sys.argv[2]) if len(sys.argv) > 2 else 60; NOF = len(sys.argv) > 3 and sys.argv[3] == "nof"; TAG = "cellfuse_nof" if NOF else "cellfuse"
EXTRA = sys.argv[4].split(",") if len(sys.argv) > 4 else []   # extra network maps from wm/out, e.g. p5
if EXTRA: TAG += "_" + "_".join(EXTRA)
INV = "/p/scratch/training2666/dasilvalopes3/inv"

def load(path, key):
    z = np.load(path); return {str(n): m for n, m in zip(z["names"], z[key])}

def feats(split):
    oc = load(f"{OUT}/occl_{meth}_{split}.npz", "maps"); cb = np.load(f"{INV}/comb/comb2_{split}.npz"); L = {str(n): m for n, m in zip(cb["names"], cb["L"])}; V = {str(n): m for n, m in zip(cb["names"], cb["V"])}
    fp = f"{INV}/F/F_{meth}_{split}.npz"; Fm = load(fp, "probs") if os.path.exists(fp) else {}
    ex = [load(f"{W}/wm/out/{meth}_{t}_{split}probs.npz", "probs") for t in EXTRA]
    names = sorted(oc); X = []
    for n in names:
        o = oc[n].astype(np.float32); o = o / (np.median(o) + 1e-6)
        l = L[n].astype(np.float32); v = V[n].astype(np.float32)
        f = Fm[n].astype(np.float32) if n in Fm else np.full((128, 128), 0.5, np.float32)
        if split == "test" and not NOF and n not in Fm: raise RuntimeError("decoder-feature map missing for a test image")
        ch = [np.clip(o, -2, 4), np.clip(l, -8, 8) / 4, np.sqrt(np.clip(v, 0, None)) / 4] + ([] if NOF else [f * 2 - 1])
        ch += [e[n].astype(np.float32) * 2 - 1 for e in ex]
        X.append(np.stack(ch))
    return names, torch.from_numpy(np.array(X))

class Net(nn.Module):
    def __init__(s, cin, w=24):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(cin, w, 5, padding=2), nn.SiLU(), nn.Conv2d(w, w, 5, padding=4, dilation=2), nn.SiLU(),
                            nn.Conv2d(w, w, 5, padding=8, dilation=4), nn.SiLU(), nn.Conv2d(w, w, 3, padding=1), nn.SiLU(), nn.Conv2d(w, 1, 1))
    def forward(s, x): return s.c(x)[:, 0]

def fit(X, Y, ep):
    X, Y = X.to(DEV), Y.to(DEV); net = Net(X.shape[1]).to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 6e-3, total_steps=ep * ((len(X) + 15) // 16))
    for e in range(ep):
        for idx in torch.randperm(len(X)).split(16):
            x, y = X[idx], Y[idx]
            if np.random.rand() < 0.5: x, y = x.flip(3), y.flip(2)
            if np.random.rand() < 0.5: x, y = x.flip(2), y.flip(1)
            lo = net(x); p = torch.sigmoid(lo)
            # BCE on the hole class (balanced) + soft IoU of the hole class
            w = torch.where(y > 0.5, 1.0, 6.0); bce = (F.binary_cross_entropy_with_logits(lo, y, reduction="none") * w).mean()
            h, ph = 1 - y, 1 - p; iou = ((h * ph).sum((1, 2)) + 1) / ((h + ph - h * ph).sum((1, 2)) + 1)
            loss = bce + 0.5 * (1 - iou).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net.eval()

names, X = feats("train"); g = gt128(); Y = torch.from_numpy(np.array([g[n] for n in names]).astype(np.float32))
print(f"== {meth}: {len(names)} whole-frame train images, features {tuple(X.shape)}; all-ones {np.mean([miou(np.ones((128,128)), y.numpy()) for y in Y]):.4f}")
rng = np.random.default_rng(0); perm = rng.permutation(len(names)); folds = np.array_split(perm, 5); P = torch.zeros_like(Y)
for k, te in enumerate(folds):
    tr = np.setdiff1d(perm, te); torch.manual_seed(k); net = fit(X[tr], Y[tr], EP)
    with torch.no_grad(): P[te] = torch.sigmoid(net(X[te].to(DEV))).cpu()
    print(f"   fold {k}: " + " ".join(f"t{t}: {np.mean([miou(p.numpy() > t, y.numpy()) for p, y in zip(P[te], Y[te])]):.3f}" for t in [0.3, 0.5, 0.7]), flush=True)
P = P.numpy(); Yn = Y.numpy()
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    print(f"   out-of-fold mIoU at threshold {t}: {np.mean([miou(p > t, y) for p, y in zip(P, Yn)]):.4f}")
occ = X[:, 0].numpy(); print(f"   reference, occlusion map alone (t 0.2/0.4): {np.mean([miou(o > 0.2, y) for o, y in zip(occ, Yn)]):.4f} / {np.mean([miou(o > 0.4, y) for o, y in zip(occ, Yn)]):.4f}")
np.savez_compressed(f"{OUT}/{TAG}_{meth}_train.npz", names=np.array(names), probs=P.astype(np.float16))
# final model on all train images -> test maps (only if all test features exist)
try:
    nt, Xt = feats("test"); torch.manual_seed(0); net = fit(X, Y, EP)
    with torch.no_grad(): Pt = torch.sigmoid(net(Xt.to(DEV))).cpu().numpy()
    np.savez_compressed(f"{OUT}/{TAG}_{meth}_test.npz", names=np.array(nt), probs=Pt.astype(np.float16)); print("   wrote test maps for", len(nt), "images")
except Exception as ex:
    print("   test maps not written:", repr(ex)[:200])
