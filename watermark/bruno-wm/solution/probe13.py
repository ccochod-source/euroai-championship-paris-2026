"""Pixel-value histogram of clean vs watermarked regions in the challenge train images (per method)."""
import json, numpy as np
from PIL import Image
from common import RELEASE
meta = json.load(open(f"{RELEASE}/train/metadata.json"))
H = {}
for m in meta:
    a = np.asarray(Image.open(f"{RELEASE}/train/images/{m['image_name']}.png").convert("RGB"))
    M = np.asarray(Image.open(f"{RELEASE}/train/masks/{m['image_name']}.png")) > 127
    for reg, sel in (("wm", M), ("clean", ~M)):
        if sel.sum() == 0: continue
        h = np.bincount(a[sel].ravel(), minlength=256).astype(np.float64)
        H[(m["method"], reg)] = H.get((m["method"], reg), 0) + h
def comb(h):
    h = h[8:248]; s = np.convolve(h, np.ones(9) / 9, mode="same"); r = (h / np.maximum(s, 1))[8:-8]   # ratio to local average
    return r
for k in sorted(H):
    r = comb(H[k]); f = np.abs(np.fft.rfft(r - r.mean())); f[0] = 0; p = int(f.argmax())
    print(k, "n=%.2e" % H[k].sum(), "ratio-to-smooth: std %.4f  min %.3f max %.3f | strongest period %.2f (amp %.2f)" % (r.std(), r.min(), r.max(), len(r) / max(p, 1), f.max()))
h = H[("trustmark", "clean")]; print("trustmark clean counts for values 100..131:", " ".join("%d" % (v / 1000) for v in h[100:132]))
h = H[("trustmark", "wm")]; print("trustmark wm    counts for values 100..131:", " ".join("%d" % (v / 1000) for v in h[100:132]))
# per-image view: fraction of pixels whose value is even / multiple of 4 etc.
for method in ("trustmark", "pixelseal", "stegastamp"):
    for reg in ("clean", "wm"):
        h = H[(method, reg)]; print(method, reg, "even %.4f  mod4==0 %.4f  mod8==0 %.4f" % (h[0::2].sum() / h.sum(), h[0::4].sum() / h.sum(), h[0::8].sum() / h.sum()))
