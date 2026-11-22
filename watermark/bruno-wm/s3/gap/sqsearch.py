"""Frame estimation, 4 margins. Model (session 75's measurement on stegastamp): the watermark was embedded on a SQUARE
slightly larger than the visible image: side 512 + e, visible image at offset (top, left). We search (e, top, left) on a
coarse grid, refine, then let the four margins move independently. Objective: slope of band-pass(image) on
band-pass(kit template computed on the reflect-padded canvas and cropped back).
usage: sqsearch.py <method> <split> [limit] [shard k/n]  -> out/frame_<method>_<split>[.kofn].json  (a=top, b=bottom, l, r)"""
import sys, json, time, numpy as np, torch
meth, split = sys.argv[1], sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SH = sys.argv[4] if len(sys.argv) > 4 else "0/1"; SK, SN = [int(v) for v in SH.split("/")]; SUF = "" if SN == 1 else f".{SK}of{SN}"
sys.argv = [sys.argv[0], meth, split]
import importlib.util
spec = importlib.util.spec_from_file_location("vs", "/p/scratch/training2666/dasilvalopes3/wm/s3/gap/vsearch_lib.py"); vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
from common import *
import os
EM = int(os.environ.get("EM", {"trustmark": 128, "pixelseal": 256, "stegastamp": 64}[meth])); CS = int(os.environ.get("CS", {"trustmark": 16, "pixelseal": 32, "stegastamp": 8}[meth]))

def search(x, bits):
    xb = vs.bp(x * 255); best = {}
    def run(cands):
        cands = [c for c in set(cands) if c not in best and min(c) >= 0 and c[0] + c[1] <= EM and c[2] + c[3] <= EM]
        if cands:
            for c, s in zip(cands, vs.scores4(x, xb, bits, cands)): best[c] = s
    sq = lambda e, t, l: (t, e - t, l, e - l)
    run([sq(e, t, l) for e in range(0, EM + 1, CS) for t in range(0, e + 1, CS) for l in range(0, e + 1, CS)])
    for step in [st for st in (16, 8, 4, 2) if st < CS]:                       # square-constrained refinement: (side, top, left)
        for _ in range(2):
            a, b, l, r = max(best, key=best.get); e = max(a + b, l + r)
            run([sq(e + d1 * step, a + d2 * step, l + d3 * step) for d1 in (-1, 0, 1) for d2 in (-1, 0, 1) for d3 in (-1, 0, 1)
                 if e + d1 * step >= 0 and 0 <= a + d2 * step <= e + d1 * step and 0 <= l + d3 * step <= e + d1 * step])
    for step in (2, 1):                          # free the four margins
        a, b, l, r = max(best, key=best.get)
        run([(a + s1 * step, b + s2 * step, l + s3 * step, r + s4 * step) for s1 in (-1, 0, 1) for s2 in (-1, 0, 1) for s3 in (-1, 0, 1) for s4 in (-1, 0, 1)])
    k = max(best, key=best.get)
    return k, best[k], best[(0, 0, 0, 0)], len(best)

wh = np.load(f"{OUT}/whole_{split}.npz"); acc = {str(n): float(v) for n, v in zip(wh["names"], wh["acc"])}
items = [it for it in load_items(split, meth) if (it["full"] if split == "train" else acc[it["name"]] > 0.70)]
if limit: items = items[:limit]
items = items[SK::SN]; res = {}; t0 = time.time()
for j, it in enumerate(items):
    x = load_img(it["path"]); bits = torch.from_numpy(it["bits"].astype(np.float32)).to("cuda")
    k, s, s0, n = search(x, bits)
    res[it["name"]] = dict(a=int(k[0]), b=int(k[1]), l=int(k[2]), r=int(k[3]), slope=float(s), slope_plain=float(s0))
    print(f"{j:3d} {it['name']} top {k[0]:2d} bottom {k[1]:2d} left {k[2]:2d} right {k[3]:2d} | extra rows {k[0] + k[1]:2d} cols {k[2] + k[3]:2d} | slope {s:+.3f} (visible frame {s0:+.3f}) | {n} frames {time.time() - t0:.0f}s", flush=True)
    if j % 10 == 9 or j == len(items) - 1: json.dump(res, open(f"{OUT}/frame_{meth}_{split}{SUF}.json", "w"))
A = np.array([[r["a"] + r["b"], r["l"] + r["r"], r["slope"], r["slope_plain"]] for r in res.values()])
print(f"== {meth} {split} n={len(A)}: slope > 0.2 on the visible frame {(A[:, 3] > 0.2).mean():.2f} -> after the frame search {(A[:, 2] > 0.2).mean():.2f} | mean slope {A[:, 3].mean():.3f} -> {A[:, 2].mean():.3f}")
print("   extra rows quantiles", np.quantile(A[:, 0], [0, .25, .5, .75, 1]).astype(int), "extra cols", np.quantile(A[:, 1], [0, .25, .5, .75, 1]).astype(int), "| corr(rows, cols)", round(float(np.corrcoef(A[:, 0], A[:, 1])[0, 1]), 2), "| mean |rows - cols|", round(float(np.abs(A[:, 0] - A[:, 1]).mean()), 1))
