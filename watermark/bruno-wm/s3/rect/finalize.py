"""Apply the acceptance rule to the raw search results, write rects_<method>_<split>.json in the agreed format
(name -> {acc_whole, rect [y0, y1, x0, x1] half-open in 512-px coordinates, acc, soft}; no 'rect' key when nothing
was verified) and, on train, report the official score on not-full images (rectangles / shapes).
CPU only. Usage: python rect/finalize.py <method> <split> <raw json>[,<raw json>...] --thr 0.75 [--variant hc|pm|pp]"""
import argparse, json
import numpy as np
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("method"); ap.add_argument("split"); ap.add_argument("raw"); ap.add_argument("--thr", type=float, default=0.75)
ap.add_argument("--variant", default=""); ap.add_argument("--write", type=int, default=1); ap.add_argument("--shrink", type=int, default=0); ap.add_argument("--shrink_below", type=int, default=100000)
ap.add_argument("--fuse", default="rect")
a = ap.parse_args()
raw = {}
for f in a.raw.split(","):
    raw.update(json.load(open(f"{OUT}/{f}")))
items = load_items(a.split, a.method)
z = np.load(f"{OUT}/whole_{a.split}.npz"); whole = {str(n): float(v) for n, v in zip(z["names"], z["acc"])}
base = {}
if a.method == "stegastamp":
    z = np.load(f"{W}/wm/out/stegastamp_ftA_{a.split}probs.npz"); base = {str(n): p.astype(np.float32) > 0.5 for n, p in zip(z["names"], z["probs"])}
out = {}
for it in items:
    r = raw.get(it["name"], {}); o = dict(acc_whole=whole[it["name"]])
    v = a.variant
    rect, acc, soft = (r.get("rect_" + v), r.get("acc_" + v), r.get("soft_" + v, r.get("soft"))) if v and ("rect_" + v) in r else (r.get("rect"), r.get("acc"), r.get("soft"))
    if rect is not None and acc >= a.thr:
        y0, y1, x0, x1 = rect
        o["frame"] = [y0, y1, x0, x1]                     # rectangle on which the decoder reads the message
        if a.shrink and min(y1 - y0, x1 - x0) < a.shrink_below and min(y1 - y0, x1 - x0) > 4 * a.shrink:
            # measured on train: the ground-truth mask is the decoded frame minus ~3 px on each side
            y0, y1, x0, x1 = y0 + a.shrink * (y0 > 0), y1 - a.shrink * (y1 < 512), x0 + a.shrink * (x0 > 0), x1 - a.shrink * (x1 < 512)
        o.update(rect=[y0, y1, x0, x1], acc=acc, soft=soft)
    out[it["name"]] = o
if a.write:
    json.dump(out, open(f"{OUT}/rects_{a.method}_{a.split}.json", "w"))
print(f"{a.method} {a.split}: {len(out)} images, searched {sum('conv' in r or 'n_hyp' in r for r in raw.values())}, rectangles accepted {sum('rect' in o for o in out.values())} (thr {a.thr}, variant '{a.variant}')")
if a.split == "train":
    G = gt128(); S = {}
    for it in items:
        gt = G[it["name"]]; o = out[it["name"]]
        k = ("full_" if it["full"] else "") + kind(gt)
        b = base.get(it["name"], np.zeros((128, 128), bool)) if not it["full"] or a.method != "stegastamp" else np.ones((128, 128), bool)
        if it["full"]:
            b = np.ones((128, 128), bool)
        pred = b
        if "rect" in o:
            m = np.zeros((512, 512), np.float32); y0, y1, x0, x1 = o["rect"]; m[y0:y1, x0:x1] = 1; pred = downsample_mask(m) > 0
            if a.fuse == "inter" and (b & pred).any():
                pred = b & pred
        S.setdefault(k, []).append((miou(b, gt), miou(pred, gt), "rect" in o, o.get("acc", 0)))
    for k, v in sorted(S.items()):
        v = np.array(v, dtype=float)
        print(f"  {k:11s} n={len(v):3d}  before {v[:, 0].mean():.3f} -> after {v[:, 1].mean():.3f}   rectangle predicted on {v[:, 2].mean():.2f}"
              + (f"   (score on those: before {v[v[:, 2] > 0, 0].mean():.3f} after {v[v[:, 2] > 0, 1].mean():.3f})" if v[:, 2].any() else ""))
