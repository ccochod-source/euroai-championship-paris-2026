"""Hypothesis H1: the challenge images were watermarked at a higher resolution S and then downscaled to 512.
Simulate on true 1024-px images and measure the same matched-filter slopes (template computed at 512) as on the
challenge images. A hypothesis fits if its slopes reproduce the REAL ones printed by probe2."""
import sys, glob, numpy as np, torch
from PIL import Image
from common import CLEAN
from wmloc.watermarks import load_watermark
from probe2 import stat
RES = {"lanczos": Image.Resampling.LANCZOS, "bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR, "box": Image.Resampling.BOX, "nearest": Image.Resampling.NEAREST}
files = sorted(glob.glob(f"{CLEAN}/flux1024/**/*.jpg", recursive=True))[:24]
REAL = {"stegastamp": "+0.040 +0.306 +0.385 +0.401 +0.452", "trustmark": "+0.037 +0.100 +0.088 +0.069 +0.092", "pixelseal": "-0.016 -0.015 +0.008 +0.032 +0.409"}
full = np.ones((512, 512), bool)
for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); rng = np.random.default_rng(0); out = {}
    for f in files:
        c = Image.open(f).convert("RGB"); msg = rng.integers(0, 2, wm.nbits)
        for S in (512, 768, 1024, 2048):
            cS = c if S == 1024 else c.resize((S, S), Image.Resampling.LANCZOS)
            wS = Image.fromarray(wm.embed(np.asarray(cS, dtype=np.uint8), msg))
            for k in (["none"] if S == 512 else ["lanczos", "bicubic", "bilinear", "box"]):
                x = np.asarray(wS if S == 512 else wS.resize((512, 512), RES[k]), dtype=np.uint8)
                T = wm.embed(x, msg).astype(np.float32) - wm.embed(x, 1 - msg).astype(np.float32)
                out.setdefault((S, k), []).append([s for s, _ in stat(x.astype(np.float32), T, full)])
    print(f"== {meth}: REAL challenge images     {REAL[meth]}")
    for (S, k), v in out.items():
        print(f"   embed at {S:4d}, downscale {k:8s} " + " ".join("%+.3f" % s for s in np.mean(v, 0)), flush=True)
