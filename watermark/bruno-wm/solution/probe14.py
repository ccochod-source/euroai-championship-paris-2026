"""Structure of the histogram comb in clean regions: per channel, per value range, per image."""
import json, numpy as np
from PIL import Image
from common import RELEASE
meta = json.load(open(f"{RELEASE}/train/metadata.json"))
Hc = np.zeros((3, 256)); Hw = np.zeros((3, 256)); per = []
for m in meta:
    a = np.asarray(Image.open(f"{RELEASE}/train/images/{m['image_name']}.png").convert("RGB"))
    M = np.asarray(Image.open(f"{RELEASE}/train/masks/{m['image_name']}.png")) > 127
    row = [m["method"], m["full_watermark"]]
    for sel, H in ((~M, Hc), (M, Hw)):
        n = int(sel.sum())
        if n < 2000: row += [np.nan, 0]; continue
        v = a[sel]
        for ch in range(3): H[ch] += np.bincount(v[:, ch], minlength=256)
        odd = (v & 1).mean(); row += [(odd - 0.5) / (0.5 / np.sqrt(v.size)), n]
    per.append(row)
print("odd fraction per channel  clean:", (Hc[:, 1::2].sum(1) / Hc.sum(1)).round(4), " wm:", (Hw[:, 1::2].sum(1) / Hw.sum(1)).round(4))
tot = Hc.sum(0); 
for lo in range(0, 256, 32):
    c = Hc[:, lo:lo + 32].sum(0); w = Hw[:, lo:lo + 32].sum(0)
    print(f"values {lo:3d}-{lo+31:3d}: odd fraction clean {c[1::2].sum()/c.sum():.4f}  wm {w[1::2].sum()/w.sum():.4f}   share of clean pixels {c.sum()/Hc.sum():.3f}")
r = Hc.sum(0); s = np.convolve(r, np.ones(5) / 5, mode="same"); ratio = r / np.maximum(s, 1)
print("clean: ratio to 5-wide local mean, values 0..47:   ", " ".join("%.2f" % x for x in ratio[:48]))
print("clean: ratio to 5-wide local mean, values 208..255:", " ".join("%.2f" % x for x in ratio[208:]))
print("clean: ratio to 5-wide local mean, values 112..143:", " ".join("%.2f" % x for x in ratio[112:144]))
zc = np.array([p[2] for p in per], float); zw = np.array([p[4] for p in per], float)
print("per-image z-score of the odd excess: clean regions: median %.1f, share>2: %.2f, share<0: %.2f | wm regions: median %.1f, share>2: %.2f" % (np.nanmedian(zc), np.nanmean(zc[~np.isnan(zc)] > 2), np.nanmean(zc[~np.isnan(zc)] < 0), np.nanmedian(zw), np.nanmean(zw[~np.isnan(zw)] > 2)))
for meth in ("pixelseal", "trustmark", "stegastamp"):
    for full in (True, False):
        sel = [i for i, p in enumerate(per) if p[0] == meth and p[1] == full]
        print(f"  {meth} full={full}: clean z median {np.nanmedian(zc[sel]):.1f} | wm z median {np.nanmedian(zw[sel]):.1f}")
