"""Exp 4 (synthetic, clean image known): can the watermark residual present in an image be predicted by re-embedding
the known message?  W = embed(C, m), true residual = W - C, template T = embed(X, m) - X.
Reports the Pearson correlation between template and true residual, and hole-detection AUCs for a composite X with a
clean square hole: (a) oracle (uses C), (b) blind high-pass correlation, (c) TrustMark removal network |X - RM(X)|.
Usage: python exp4_fidelity.py --kit . --clean /p/scratch/training2666/dasilvalopes3/clean/val2017 --n 16
"""
import argparse, glob, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True); ap.add_argument("--clean", required=True); ap.add_argument("--n", type=int, default=16)
ap.add_argument("--methods", default="trustmark,pixelseal,stegastamp")
a = ap.parse_args()
sys.path.insert(0, a.kit)
from wmloc.watermarks import load_watermark  # noqa


def load(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((512, 512), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def corr(x, y):
    x = x.ravel() - x.mean(); y = y.ravel() - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def auc(score, label):
    label = label.astype(bool).ravel(); score = score.ravel(); n1, n0 = label.sum(), (~label).sum()
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def local_corr(A, B, k):
    s = cv2.blur((A * B).sum(2), (k, k))
    return s / (np.sqrt(cv2.blur((A * A).sum(2), (k, k)) * cv2.blur((B * B).sum(2), (k, k))) + 1e-6)


def to128(m):
    return m.reshape(128, 4, 128, 4).mean((1, 3))


files = sorted(glob.glob(f"{a.clean}/*.jpg"))[:a.n]
imgs = [load(p) for p in files]
rng = np.random.default_rng(0)
hole = np.zeros((512, 512), bool); hole[192:320, 160:352] = True          # clean region in the composite
lab128 = to128((~hole).astype(np.float32)) > 0.5                          # 1 = watermarked
for meth in a.methods.split(","):
    wm = load_watermark(meth, models_dir=Path(a.kit) / "models", device="cuda")
    res = {k: [] for k in ("psnr", "c_W", "c_X", "c_ctrl", "c_low", "auc_oracle17", "auc_hp17", "auc_hp33", "auc_rm", "auc_rmT")}
    for C in imgs:
        m = rng.integers(0, 2, wm.nbits); m2 = rng.integers(0, 2, wm.nbits)
        Cf = C.astype(np.float32)
        W = wm.embed(C, m); Rt = W.astype(np.float32) - Cf
        res["psnr"].append(10 * np.log10(255 ** 2 / (Rt ** 2).mean()))
        RW = wm.embed(W, m).astype(np.float32) - W
        res["c_W"].append(corr(Rt, RW))
        res["c_ctrl"].append(corr(Rt, wm.embed(W, m2).astype(np.float32) - W))
        X = np.where(hole[..., None], C, W); Xf = X.astype(np.float32)
        T = wm.embed(X, m).astype(np.float32) - Xf
        res["c_X"].append(corr(Rt[~hole], T[~hole]))
        res["c_low"].append(corr(cv2.GaussianBlur(Rt, (0, 0), 2.0), cv2.GaussianBlur(T, (0, 0), 2.0)))
        res["auc_oracle17"].append(auc(to128(local_corr(Xf - Cf, T, 17)), lab128))
        HP = Xf - cv2.GaussianBlur(Xf, (0, 0), 1.5)
        res["auc_hp17"].append(auc(to128(local_corr(HP, T, 17)), lab128))
        res["auc_hp33"].append(auc(to128(local_corr(HP, T, 33)), lab128))
        if meth == "trustmark":
            try:
                RM = np.asarray(wm.tm.remove_watermark(Image.fromarray(X)).convert("RGB"), dtype=np.float32)
                E = Xf - RM
                res["auc_rm"].append(auc(to128(cv2.blur(np.abs(E).sum(2), (17, 17))), lab128))
                res["auc_rmT"].append(auc(to128(local_corr(E, T, 17)), lab128))
                res.setdefault("c_rm", []).append(corr(E[~hole], Rt[~hole]))
            except Exception as e:
                res.setdefault("rm_err", []).append(repr(e)[:200])
    print(f"\n===== {meth} (n={len(imgs)})")
    for k, v in res.items():
        if v and isinstance(v[0], str):
            print(f"  {k}: {v[0]}")
        elif v:
            print(f"  {k:13s} mean={np.mean(v):.3f}  min={np.min(v):.3f}")
    del wm; torch.cuda.empty_cache()
print("""
c_W   : corr(true residual, re-embedding the watermarked image with the same message)
c_ctrl: same with a different message (how much of the residual is message-independent)
c_X   : corr on the watermarked part of a composite with a clean hole
auc_oracle17: hole detection AUC if the clean image were known (upper bound for the template)
auc_hp*: blind high-pass correlation with the template ; auc_rm: TrustMark removal residual magnitude ; auc_rmT: removal residual x template""")
