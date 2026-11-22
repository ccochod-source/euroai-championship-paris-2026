"""Learned refinement: small conv net over stacked maps (per-image z-scored) + method one-hot -> 128x128 mask.
5-fold CV on train to report honest mIoU; then train on all train and predict test.
Usage: python refine.py --train maps_train2.npz [--test maps_test2.npz --out output/sub_refine.pt [--submit]]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--test"); ap.add_argument("--out")
ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--submit", action="store_true")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
MAPS = ["pp", "win128", "win192", "win256", "grad"]
METH = ["pixelseal", "stegastamp", "trustmark"]


def feats(d):
    xs = []
    for k in MAPS:
        M = np.nan_to_num(d["map_" + k].astype(np.float32), nan=0.0)
        M = (M - M.mean((1, 2), keepdims=True)) / (M.std((1, 2), keepdims=True) + 1e-6)
        xs.append(M)
    oh = np.stack([(d["method"] == m).astype(np.float32) for m in METH], 1)[:, :, None, None] * np.ones((1, 1, 128, 128), np.float32)
    return np.concatenate([np.stack(xs, 1), oh], 1)


def block(i, o, dil=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=dil, dilation=dil), nn.BatchNorm2d(o), nn.ReLU())


class Net(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(block(c, 32), block(32, 32, 2), block(32, 32, 4), block(32, 32, 8), block(32, 32, 16),
                               block(32, 32, 1))
        self.h = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        return self.h(self.b(x))[:, 0]


def soft_miou_loss(lg, y):
    p = torch.sigmoid(lg)
    i1 = (p * y).sum((1, 2)); u1 = (p + y - p * y).sum((1, 2))
    i0 = ((1 - p) * (1 - y)).sum((1, 2)); u0 = ((1 - p) + (1 - y) - (1 - p) * (1 - y)).sum((1, 2))
    return 1 - 0.5 * (i1 / (u1 + 1) + i0 / (u0 + 1)).mean()


def aug(x, y):
    if np.random.rand() < .5: x, y = x.flip(-1), y.flip(-1)
    if np.random.rand() < .5: x, y = x.flip(-2), y.flip(-2)
    k = np.random.randint(4)
    return torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))


def train(X, Y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(X.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y).float().to(dev)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 32):
            xb, yb = aug(Xt[perm[i:i + 32]], Yt[perm[i:i + 32]])
            lg = net(xb)
            loss = F.binary_cross_entropy_with_logits(lg, yb) + soft_miou_loss(lg, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net.eval()


@torch.no_grad()
def predict(nets, X):
    Xt = torch.from_numpy(X).to(dev)
    out = 0
    for n in nets:
        for k in range(4):  # TTA rot90
            out = out + torch.rot90(torch.sigmoid(n(torch.rot90(Xt, k, (-2, -1)))), -k, (-2, -1))
    return (out / (4 * len(nets))).cpu().numpy()


def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = np.logical_or(p == c, g == c).sum()
        s += 1.0 if u == 0 else np.logical_and(p == c, g == c).sum() / u
    return s / 2


tr = np.load(a.train); X, Y, meth = feats(tr), tr["gt"].astype(np.float32), tr["method"]
rng = np.random.RandomState(0); folds = rng.permutation(len(X)) % 5
P = np.zeros_like(Y)
for f in range(5):
    nets = [train(X[folds != f], Y[folds != f], s) for s in range(a.seeds)]
    P[folds == f] = predict(nets, X[folds == f])
    print("fold", f, "done", flush=True)
for t in (0.4, 0.5, 0.6):
    sc = np.array([miou((p > t).astype(np.uint8), g) for p, g in zip(P, Y)])
    print(f"CV t={t}: overall {sc.mean():.4f} | " + " ".join(f"{m}={sc[meth == m].mean():.4f}" for m in METH))

if a.test:
    nets = [train(X, Y, s) for s in range(a.seeds)]
    te = np.load(a.test); Pt = predict(nets, feats(te))
    names = [str(n) for n in te["names"]]
    masks = (Pt > 0.5).astype(np.uint8)
    torch.save({"image_names": names, "masks": torch.from_numpy(masks).bool()}, a.out)
    print("wrote", a.out, masks.shape, "pos frac", masks.mean().round(3))
    if a.submit:
        import requests
        api = (Path.home() / ".hack_api_key").read_text().strip()
        with open(a.out, "rb") as fh:
            r = requests.post("http://34.147.229.55/submit/34-watermark-localisation-image", headers={"X-API-Key": api},
                              files={"file": (Path(a.out).name, fh, "application/octet-stream")}, timeout=(10, 600))
        print(r.status_code, r.text[:500])
