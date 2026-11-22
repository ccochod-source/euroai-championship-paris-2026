"""Pyramid frame search (fast): square frame of side 512 + e, visible window at (top, left).
Stage 1: coarse grid scored on a coarse band (broad peak); stage 2-3: refine on finer bands with halving steps;
last: free the four margins by +-1. Single-message template difference m vs 1-m.
usage: pyr.py <method> <split> [limit] [shard k/n] [which: full|notfull|all] -> out/frame_<method>_<split>[.kofn].json"""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
meth, split = sys.argv[1], sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SH = sys.argv[4] if len(sys.argv) > 4 else "0/1"; SK, SN = [int(v) for v in SH.split("/")]; SUF = "" if SN == 1 else f".{SK}of{SN}"
WHICH = sys.argv[5] if len(sys.argv) > 5 else "full"
sys.argv = [sys.argv[0], meth, split]
import importlib.util
spec = importlib.util.spec_from_file_location("vs", "/p/scratch/training2666/dasilvalopes3/wm/s3/gap/vsearch_lib.py"); vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
from common import *
wm = vs.wm; dev = "cuda"
EM = {"trustmark": 144, "pixelseal": 288, "stegastamp": 64}[meth]
# (band lo, band hi, step) per stage; the first stage is a full grid, the others are local refinements
STAGES = {"trustmark": [(6, 12, 24), (3, 6, 12), (3, 6, 6), (3, 6, 3), (3, 6, 1)],
          "pixelseal": [(4, 8, 32), (3, 6, 16), (1.5, 4, 8), (1, 3, 4), (1, 3, 2), (1, 3, 1)],
          "stegastamp": [(6, 12, 8), (3, 6, 4), (3, 6, 2), (3, 6, 1)]}[meth]
NAT = {"trustmark": getattr(getattr(wm, "tm", None), "model_resolution_enc", 256), "stegastamp": 400, "pixelseal": 256}[meth]
MODE = "bicubic" if meth == "stegastamp" else "bilinear"

def bpb(x, lo, hi): return vs.blur(x, lo) - vs.blur(x, hi)

@torch.no_grad()
def score(x, xbs, bits, hm, cands, band, bs=48):
    lo, hi = band; xb = xbs[band]; out = []
    for i in range(0, len(cands), bs):
        cs = cands[i:i + bs]
        cv = torch.cat([F.interpolate(vs.canvas4(x, *c), size=(NAT, NAT), mode=MODE, align_corners=False, antialias=True) for c in cs]).clamp(0, 1)
        res = vs.native_residual(cv, bits) - vs.native_residual(cv, 1 - bits)
        for (a, b, l, r), rr in zip(cs, res):
            T = F.interpolate(rr[None], size=(512 + a + b, 512 + l + r), mode="bilinear", align_corners=False, antialias=(meth == "pixelseal"))[:, :, a:a + 512, l:l + 512]
            if hm is not None: T = T * hm
            if T.shape[1] == 1: T = T.repeat(1, 3, 1, 1)
            tb = bpb(T, lo, hi); out.append(float((xb * tb).sum() / (tb * tb).sum().clamp_min(1e-6)))
    return np.array(out)

def search(x, bits):
    hm = wm.model.attenuation.heatmaps(x) if (meth == "pixelseal" and wm.model.attenuation is not None) else None
    bands = sorted(set((lo, hi) for lo, hi, _ in STAGES)); xbs = {b: bpb(x * 255, *b) for b in bands}
    sq = lambda e, t, l: (t, e - t, l, e - l); ok = lambda c: min(c) >= 0 and c[0] + c[1] <= EM + 32 and c[2] + c[3] <= EM + 32
    lo, hi, st = STAGES[0]; cands = [sq(e, t, l) for e in range(0, EM + 1, st) for t in range(0, e + 1, st) for l in range(0, e + 1, st)]
    sc = score(x, xbs, bits, hm, cands, (lo, hi)); n = len(cands)
    tops = [cands[i] for i in np.argsort(-sc)[:3]]
    for lo, hi, st in STAGES[1:]:
        cc = sorted(set(c for (a, b, l, r) in tops for e in [max(a + b, l + r)] for c in
                        [sq(e + d1 * st, a + d2 * st, l + d3 * st) for d1 in (-1, 0, 1) for d2 in (-1, 0, 1) for d3 in (-1, 0, 1)
                         if e + d1 * st >= 0 and 0 <= a + d2 * st <= e + d1 * st and 0 <= l + d3 * st <= e + d1 * st] if ok(c)))
        sc = score(x, xbs, bits, hm, cc, (lo, hi)); n += len(cc); tops = [cc[i] for i in np.argsort(-sc)[:2]]
    a, b, l, r = tops[0]; fin = STAGES[-1][:2]
    cc = [c for c in set((a + s1, b + s2, l + s3, r + s4) for s1 in (-1, 0, 1) for s2 in (-1, 0, 1) for s3 in (-1, 0, 1) for s4 in (-1, 0, 1)) if ok(c)] + [(0, 0, 0, 0)]
    sc = score(x, xbs, bits, hm, cc, fin); k = int(np.argmax(sc[:-1]))
    return cc[k], float(sc[k]), float(sc[-1]), n + len(cc)

wh = np.load(f"{OUT}/whole_{split}.npz"); acc = {str(n): float(v) for n, v in zip(wh["names"], wh["acc"])}
isfull = lambda it: (it["full"] if split == "train" else acc[it["name"]] > 0.70)
items = [it for it in load_items(split, meth) if WHICH == "all" or (isfull(it) == (WHICH == "full"))]
if limit: items = items[:limit]
items = items[SK::SN]; res = {}; t0 = time.time(); tag = "" if WHICH == "full" else f"_{WHICH}"
for j, it in enumerate(items):
    x = load_img(it["path"]); bits = torch.from_numpy(it["bits"].astype(np.float32)).to(dev)
    k, s, s0, n = search(x, bits)
    res[it["name"]] = dict(a=int(k[0]), b=int(k[1]), l=int(k[2]), r=int(k[3]), slope=s, slope_plain=s0)
    print(f"{j:3d} {it['name']} top {k[0]:3d} bottom {k[1]:3d} left {k[2]:3d} right {k[3]:3d} | extra rows {k[0] + k[1]:3d} cols {k[2] + k[3]:3d} | slope {s:+.3f} (visible frame {s0:+.3f}) | {n} frames {time.time() - t0:.0f}s", flush=True)
    if j % 10 == 9 or j == len(items) - 1: json.dump(res, open(f"{OUT}/frame{tag}_{meth}_{split}{SUF}.json", "w"))
A = np.array([[r["a"] + r["b"], r["l"] + r["r"], r["slope"], r["slope_plain"]] for r in res.values()])
print(f"== {meth} {split} {WHICH} n={len(A)}: slope > 0.2 on the visible frame {(A[:, 3] > 0.2).mean():.2f} -> after the frame search {(A[:, 2] > 0.2).mean():.2f} | mean slope {A[:, 3].mean():.3f} -> {A[:, 2].mean():.3f}")
print("   extra rows quantiles", np.quantile(A[:, 0], [0, .25, .5, .75, 1]).astype(int), "extra cols", np.quantile(A[:, 1], [0, .25, .5, .75, 1]).astype(int), "| mean |rows - cols|", round(float(np.abs(A[:, 0] - A[:, 1]).mean()), 1))
