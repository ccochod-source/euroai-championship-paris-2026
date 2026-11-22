"""Exp 8: which generation pipeline reproduces the statistics of the REAL fully-watermarked images?
P1 direct: embed at 512.  P2/P3: embed at 1024 then downscale to 512 (LANCZOS / BICUBIC / AREA).
Statistic: slope of HP(image) on HP(template) in watermarked minus clean area (real masks), plus decoder margin.
Reference (real images): trustmark -0.10, pixelseal -0.03, stegastamp +0.62 ; margins 7.62 / 8.83 / 6.01."""
import argparse, glob, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True); ap.add_argument("--data", required=True); ap.add_argument("--flux", required=True)
ap.add_argument("--n", type=int, default=24); ap.add_argument("--methods", default="trustmark,pixelseal,stegastamp")
a = ap.parse_args(); sys.path.insert(0, a.kit)
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor

def hp(x, s=1.5): return x - cv2.GaussianBlur(x, (0, 0), s)
def slope(X, T, m):
    H, G = hp(X.astype(np.float32)), hp(T)
    return float((H[m] * G[m]).sum() / ((G[m] ** 2).sum() + 1e-9))
def margin(wm, X, msg):
    lg = wm.logits(to_tensor(X))[0].float().cpu().numpy(); return float((lg * (msg * 2.0 - 1)).mean())
def down(x, how):
    if how == "area": return cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(x).resize((512, 512), {"lanczos": Image.Resampling.LANCZOS, "bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}[how]))

files = sorted(glob.glob(f"{a.flux}/**/*.jpg", recursive=True) + glob.glob(f"{a.flux}/**/*.png", recursive=True))[:a.n]
big = [np.asarray(Image.open(p).convert("RGB")) for p in files]
print("flux images:", len(big), big[0].shape)
rng = np.random.default_rng(0)
for meth in a.methods.split(","):
    wm = load_watermark(meth, models_dir=Path(a.kit) / "models", device="cuda")
    ds = LocalisationDataset(a.data, "train", methods=[meth])
    masks = [ds[i]["mask"].astype(bool) for i, m in enumerate(ds.meta) if m.get("full_watermark", False)]
    masks = [m for m in masks if (~m).sum() > 2000 and not m.all()][:len(big)]
    res = {}
    for k, (B, mask) in enumerate(zip(big, masks)):
        msg = rng.integers(0, 2, wm.nbits)
        if B.shape[0] != 1024:
            B = np.asarray(Image.fromarray(B).resize((1024, 1024), Image.Resampling.LANCZOS))
        WB = wm.embed(B, msg)
        for name in ("P1 direct 512", "P2 1024->lanczos", "P3 1024->bicubic", "P3 1024->area", "P3 1024->bilinear"):
            how = name.split("->")[1] if "->" in name else "lanczos"
            C = down(B, how)
            W = wm.embed(C, msg) if name.startswith("P1") else down(WB, how)
            X = np.where(mask[..., None], W, C)
            T = wm.embed(X, msg).astype(np.float32) - X
            res.setdefault(name, []).append((slope(X, T, mask) - slope(X, T, ~mask), margin(wm, X, msg), 10 * np.log10(255 ** 2 / max(1e-9, ((W.astype(np.float32) - C) ** 2).mean()))))
    print(f"\n===== {meth}")
    for name, v in res.items():
        v = np.array(v); print(f"  {name:18s} slope diff {v[:,0].mean():+.3f}   margin {v[:,1].mean():5.2f}   PSNR {v[:,2].mean():.1f} dB")
    del wm; torch.cuda.empty_cache()
