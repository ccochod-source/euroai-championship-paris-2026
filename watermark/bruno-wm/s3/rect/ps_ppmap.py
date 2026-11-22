"""PixelSeal experiment: per-position agreement map (local-GRN convolutional mode) around a verified frame, as a
SEGMENTATION of where the watermark is (edges of rectangles, object cut-outs of shape images).
The image is resized so that the frame covers K cells per side; 16 sub-cell shifts give a map 4x finer (1 fine cell
= frame side / (4K)). Map value = bit accuracy of the per-position logits against the known message.
Usage: sg 2 rect/ps_ppmap.py --inp rect_raw_pixelseal_train.json [--K 10]"""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from common import *
import pslib

torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True); ap.add_argument("--K", type=int, default=10); ap.add_argument("--min_acc", type=float, default=0.70)
ap.add_argument("--split", default="train"); ap.add_argument("--save", default="")
a = ap.parse_args()
wm = pslib.patch(load_wm("pixelseal"))
raw = json.load(open(f"{OUT}/{a.inp}")); items = {it["name"]: it for it in load_items(a.split, "pixelseal")}; G = gt128() if a.split == "train" else {}


@torch.no_grad()
def ppmap(img, sm, rect, K):
    """512 x 512 map of per-position bit accuracy (nan where not covered)."""
    y0, y1, x0, x1 = rect; Hf = int(np.clip(round(512 * K / (y1 - y0)), 8, 128)); Wf = int(np.clip(round(512 * K / (x1 - x0)), 8, 128))
    x = F.interpolate(img, size=(Hf * 32, Wf * 32), mode="bilinear", align_corners=False, antialias=True)
    fine = torch.full((4 * Hf, 4 * Wf), float("nan"), device=img.device)
    for k in range(4):
        for l in range(4):
            xs = F.pad(x, (8 * l, 0, 8 * k, 0), mode="replicate")[:, :, :Hf * 32, :Wf * 32]
            L = pslib.ppl(wm, xs, (9, 9), amp=False)[0]
            acc = ((L > 0).float() * 2 - 1 == sm[:, None, None]).float().mean(0)          # Hf x Wf
            i0, j0 = (1 if k else 0), (1 if l else 0)
            fine[4 * i0 - k::4, 4 * j0 - l::4][:Hf - i0, :Wf - j0] = acc[i0:, j0:]
    # fine cell f covers resized px [8f, 8f + 32): centre 8f + 16 -> shift the map by 1.5 fine cells when resampling
    m = torch.nan_to_num(fine, nan=0.5)[None, None]
    m = F.pad(m, (2, 0, 2, 0), mode="replicate")[:, :, :4 * Hf, :4 * Wf]                    # centre correction (~1.5 -> 2 cells)
    return F.interpolate(m, size=(512, 512), mode="bilinear", align_corners=False)[0, 0]


rows, t0 = [], time.time()
for name, r in raw.items():
    if r.get("acc", 0) < a.min_acc or r.get("full"):
        continue
    it = items[name]; img = load_img(it["path"]); sm = sm_of(it["bits"]); y0, y1, x0, x1 = r["rect"]
    M = ppmap(img, sm, r["rect"], a.K)
    Ms = F.avg_pool2d(M[None, None], 9, stride=1, padding=4)[0, 0]
    inside = Ms[y0:y1, x0:x1]; lvl = float(inside.flatten().kthvalue(int(0.7 * inside.numel()) + 1).values)   # 70% quantile inside the frame
    row = dict(name=name, kind=r.get("kind"), lvl=lvl)
    if name in G:
        gt = G[name]; rm = np.zeros((512, 512), np.float32); rm[y0:y1, x0:x1] = 1; row["rect"] = miou(downsample_mask(rm) > 0, gt); row["zeros"] = miou(np.zeros_like(gt), gt)
        gy0, gy1, gx0, gx1 = max(0, y0 - (y1 - y0) // 3), min(512, y1 + (y1 - y0) // 3), max(0, x0 - (x1 - x0) // 3), min(512, x1 + (x1 - x0) // 3)
        win = torch.zeros_like(Ms, dtype=torch.bool); win[gy0:gy1, gx0:gx1] = True               # only look near the frame
        for f in (0.3, 0.4, 0.5, 0.6):
            thr = 0.5 + f * (lvl - 0.5); pm = ((Ms > thr) & win).float().cpu().numpy(); p128 = downsample_mask(pm) > 0
            row[f"map{f}"] = miou(p128, gt)
            ys, xs = np.nonzero(pm)
            if len(ys):
                bm = np.zeros((512, 512), np.float32); bm[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1; row[f"box{f}"] = miou(downsample_mask(bm) > 0, gt)
            else:
                row[f"box{f}"] = row["zeros"]
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
print("DONE", len(rows), f"{time.time() - t0:.0f}s")
for kd in ("rect", "shape"):
    R = [r for r in rows if r["kind"] == kd]
    if R:
        print(kd, "n", len(R), " ".join(f"{k} {np.mean([r[k] for r in R]):.3f}" for k in R[0] if k not in ("name", "kind", "lvl")))
