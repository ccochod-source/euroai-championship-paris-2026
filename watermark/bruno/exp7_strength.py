"""Exp 7: is the watermark in the REAL fully-watermarked images as strong as the kit embedder's output at 512 px?
Compares decoder margin / bit accuracy, and the regression slope of the image high-pass on the template high-pass
(watermarked area minus clean area), between real images and synthetic composites built with the same masks."""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True); ap.add_argument("--data", required=True); ap.add_argument("--syn", required=True)
ap.add_argument("--n", type=int, default=40); ap.add_argument("--methods", default="stegastamp,trustmark,pixelseal")
a = ap.parse_args(); sys.path.insert(0, a.kit)
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor

def hp(x, s=1.5): return x - cv2.GaussianBlur(x, (0, 0), s)
def slope(X, T, m):
    H, G = hp(X.astype(np.float32)), hp(T)
    return float((H[m] * G[m]).sum() / ((G[m] ** 2).sum() + 1e-9))
def dec(wm, X, msg):
    lg = wm.logits(to_tensor(X))[0].float().cpu().numpy(); s = msg * 2.0 - 1
    return float(((lg > 0) == (msg > 0)).mean()), float((lg * s).mean()), float(np.abs(lg).mean())

for meth in a.methods.split(","):
    wm = load_watermark(meth, models_dir=Path(a.kit) / "models", device="cuda")
    ds = LocalisationDataset(a.data, "train", methods=[meth])
    idx = [i for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:a.n]
    clean = np.load(f"{a.syn}/clean.npy", mmap_mode="r"); full = np.load(f"{a.syn}/{meth}_full_a.npy", mmap_mode="r"); msgs = np.load(f"{a.syn}/{meth}_msgs_a.npy")
    R, S = [], []
    for k, i in enumerate(idx):
        it = ds[i]; X = it["image"]; mask = it["mask"].astype(bool)
        if mask.all() or (~mask).sum() < 2000: continue
        T = wm.embed(X, it["message"]).astype(np.float32) - X
        R.append(dec(wm, X, it["message"]) + (slope(X, T, mask), slope(X, T, ~mask), float(np.abs(T).mean()), float(X.std())))
        j = 200 + k; Xs = np.where(mask[..., None], np.array(full[j]), np.array(clean[j]))
        Ts = wm.embed(Xs, msgs[j, 0]).astype(np.float32) - Xs
        S.append(dec(wm, Xs, msgs[j, 0]) + (slope(Xs, Ts, mask), slope(Xs, Ts, ~mask), float(np.abs(Ts).mean()), float(Xs.std())))
    R, S = np.array(R), np.array(S)
    print(f"\n===== {meth} (n={len(R)})   columns: bit acc | margin | |logit| | slope wm | slope clean | slope diff | |T| | image std")
    for nm, M in (("REAL     ", R), ("SYNTHETIC", S)):
        print(f"  {nm}: {M[:,0].mean():.3f} | {M[:,1].mean():6.2f} | {M[:,2].mean():6.2f} | {M[:,3].mean():+.3f} | {M[:,4].mean():+.3f} | {np.mean(M[:,3]-M[:,4]):+.3f} | {M[:,5].mean():.2f} | {M[:,6].mean():.1f}")
    del wm; torch.cuda.empty_cache()
