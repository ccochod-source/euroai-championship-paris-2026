"""Probe PixelSeal on the real train not-full images: decodability at the ground-truth rectangle vs size,
interpolation mode, tolerance to growth, and agreement between convolutional window scoring and crop scoring."""
import time, sys
import numpy as np, torch, torch.nn.functional as F
from common import *

wm = load_wm("pixelseal"); det = wm.det
print(torch.cuda.get_device_name(0))
print(type(det).__name__, [n for n, _ in det.named_children()])
print("convnext:", type(det.convnext).__name__, [n for n, _ in det.convnext.named_children()])
print("pixel_decoder:", det.pixel_decoder)
print("has GRN:", any("grn" in n.lower() for n, _ in det.named_modules()))
items = [it for it in load_items("train", "pixelseal") if not it["full"]]
G = gt128()
x = torch.rand(256, 3, 256, 256, device="cuda")
with torch.no_grad():
    det(x[:8]); torch.cuda.synchronize(); t = time.time(); det(x); torch.cuda.synchronize()
print("crops/s", 256 / (time.time() - t), "mem GB", torch.cuda.max_memory_allocated() / 1e9)


def crop(img, r, mode="bilinear"):
    y0, y1, x0, x1 = r
    return F.interpolate(img[:, :, y0:y1, x0:x1], size=(256, 256), mode=mode, align_corners=False, antialias=True)


def lg(x):
    with torch.no_grad():
        return det(x)[:, 1:].float()


def grow(r, g):
    y0, y1, x0, x1 = r; h, w = y1 - y0, x1 - x0
    dy, dx = int(round(h * (g - 1) / 2)), int(round(w * (g - 1) / 2))
    return (max(0, y0 - dy), min(512, y1 + dy), max(0, x0 - dx), min(512, x1 + dx))


rows = []
for it in items:
    k = kind(G[it["name"]])
    if k != "rect":
        continue
    m = gt512(it["name"]); ys, xs = np.nonzero(m)
    r = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    img = load_img(it["path"]); sm = sm_of(it["bits"])
    out = dict(name=it["name"], h=r[1] - r[0], w=r[3] - r[2])
    for mode in ["bilinear", "bicubic", "nearest-exact"]:
        l = lg(crop(img, r, mode) if mode != "nearest-exact" else F.interpolate(img[:, :, r[0]:r[1], r[2]:r[3]], size=(256, 256), mode=mode))
        out["acc_" + mode] = float(((l > 0).float() * 2 - 1 == sm).float().mean()); out["mar_" + mode] = float((l * sm).mean())
    for g in [1.1, 1.2, 1.35, 1.5]:
        l = lg(crop(img, grow(r, g)))
        out[f"acc_g{g}"] = float(((l > 0).float() * 2 - 1 == sm).float().mean())
    rows.append(out)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
import json
json.dump(rows, open(f"{OUT}/rect_ps_probe.json", "w"))
a = lambda k: np.array([r[k] for r in rows])
short = np.minimum(a("h"), a("w"))
print("n rect", len(rows))
for k in rows[0]:
    if k.startswith("acc"):
        v = a(k); print(k, "mean %.3f  >0.60 %.2f  >0.65 %.2f  >0.70 %.2f  >0.75 %.2f" % (v.mean(), (v > .6).mean(), (v > .65).mean(), (v > .7).mean(), (v > .75).mean()))
for lo, hi in [(0, 24), (24, 40), (40, 64), (64, 100), (100, 160), (160, 600)]:
    s = (short >= lo) & (short < hi)
    if s.any():
        print(f"short side [{lo},{hi}) n={s.sum()} acc_bilinear mean {a('acc_bilinear')[s].mean():.3f} >0.65: {(a('acc_bilinear')[s] > .65).mean():.2f}  g1.2: {a('acc_g1.2')[s].mean():.3f} g1.5 {a('acc_g1.5')[s].mean():.3f}")
