"""Exp 5: is the residual in the REAL fully-watermarked images the one the kit embedder predicts for the known message?
Statistic per image (uses the GT mask): corr(HP(X), T) over watermarked pixels (z_wm) vs over clean pixels (z_hole),
with T = embed(X, m) - X and HP(X) = X - blur(X). Same statistic on synthetic composites (clean image known) built with
the same real masks, as the reference of what a matching template gives.
Usage: python exp5_realcheck.py --kit . --data release --syn /p/scratch/training2666/dasilvalopes3/syn --n 40
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True); ap.add_argument("--data", required=True); ap.add_argument("--syn", required=True)
ap.add_argument("--n", type=int, default=40); ap.add_argument("--methods", default="trustmark,pixelseal,stegastamp")
a = ap.parse_args()
sys.path.insert(0, a.kit)
from wmloc.data import LocalisationDataset  # noqa
from wmloc.watermarks import load_watermark  # noqa


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def zstats(X, T, mask):
    Xf = X.astype(np.float32); out = []
    for sig in (0.8, 1.5, 3.0):
        HP = Xf - cv2.GaussianBlur(Xf, (0, 0), sig)
        out += [corr(HP[mask], T[mask]), corr(HP[~mask], T[~mask])]
    return out


for meth in a.methods.split(","):
    wm = load_watermark(meth, models_dir=Path(a.kit) / "models", device="cuda")
    ds = LocalisationDataset(a.data, "train", methods=[meth])
    idx = [i for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:a.n]
    clean = np.load(f"{a.syn}/clean.npy", mmap_mode="r"); full = np.load(f"{a.syn}/{meth}_full_a.npy", mmap_mode="r")
    msgs = np.load(f"{a.syn}/{meth}_msgs_a.npy")
    real, syn, wrong = [], [], []
    rng = np.random.default_rng(0)
    for k, i in enumerate(idx):
        it = ds[i]; X = it["image"]; mask = it["mask"].astype(bool)
        if mask.all() or (~mask).sum() < 2000:
            continue
        T = wm.embed(X, it["message"]).astype(np.float32) - X
        real.append(zstats(X, T, mask))
        Tw = wm.embed(X, rng.integers(0, 2, wm.nbits)).astype(np.float32) - X          # wrong message: control
        wrong.append(zstats(X, Tw, mask))
        j = 100 + k
        Xs = np.where(mask[..., None], np.array(full[j]), np.array(clean[j]))
        Ts = wm.embed(Xs, msgs[j, 0]).astype(np.float32) - Xs
        syn.append(zstats(Xs, Ts, mask))
    R, S, Wr = np.array(real), np.array(syn), np.array(wrong)
    print(f"\n===== {meth}: n={len(R)} fully-watermarked images with a clean region")
    for c, sig in enumerate((0.8, 1.5, 3.0)):
        for nm, M in (("REAL, true message ", R), ("REAL, wrong message", Wr), ("SYNTHETIC          ", S)):
            zw, zh = M[:, 2 * c], M[:, 2 * c + 1]
            print(f"  blur {sig}: {nm} corr on watermarked area={zw.mean():+.4f}  on clean area={zh.mean():+.4f}  "
                  f"diff={np.mean(zw - zh):+.4f}  images with wm>clean: {np.mean(zw > zh):.2f}")
    del wm; torch.cuda.empty_cache()
