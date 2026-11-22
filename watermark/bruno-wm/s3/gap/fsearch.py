"""Fast frame estimation. Model: the watermark was embedded on a square of side 512 + e that contains the visible image
at offset (top, left). For each e we compute ONE kit template on the centred reflect-padded canvas and find the offset by
FFT cross-correlation with the band-passed image (the pattern is anchored to the frame, so it barely depends on where the
window sits); the best (e, top, left) is then refined with exact templates and the four margins are freed.
Score = regression slope of band-pass(image) on band-pass(template) (~0.3-0.5 when the frame is right, ~0 otherwise).
usage: fsearch.py <method> <split> [limit] [shard k/n] [which: full|notfull|all]
 -> out/frame_<method>_<split>[.kofn].json : name -> a (top), b (bottom), l, r, slope, slope_plain, e_curve"""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
meth, split = sys.argv[1], sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SH = sys.argv[4] if len(sys.argv) > 4 else "0/1"; SK, SN = [int(v) for v in SH.split("/")]; SUF = "" if SN == 1 else f".{SK}of{SN}"
WHICH = sys.argv[5] if len(sys.argv) > 5 else "full"
sys.argv = [sys.argv[0], meth, split]
import importlib.util
spec = importlib.util.spec_from_file_location("vs", "/p/scratch/training2666/dasilvalopes3/wm/s3/gap/vsearch_lib.py"); vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
from common import *
wm = vs.wm; dev = "cuda"
EMAX = {"trustmark": 192, "pixelseal": 384, "stegastamp": 96}[meth]; ESTEP = 8 if meth != "stegastamp" else 4
NAT = {"trustmark": getattr(getattr(wm, "tm", None), "model_resolution_enc", 256), "stegastamp": 400, "pixelseal": 256}[meth]
MODE = "bicubic" if meth == "stegastamp" else "bilinear"

@torch.no_grad()
def coarse(x, xb, bits, hm):
    """for each e: best offset by FFT correlation. returns list of (slope, e, top, left)."""
    out = []; w = hm if hm is not None else torch.ones_like(x[:, :1]); xw = xb * w; w2 = (w * w).expand(-1, 3, -1, -1)
    es = list(range(0, EMAX + 1, ESTEP))
    for i in range(0, len(es), 16):
        ee = es[i:i + 16]
        cv = torch.cat([F.interpolate(vs.canvas4(x, e // 2, e - e // 2, e // 2, e - e // 2), size=(NAT, NAT), mode=MODE, align_corners=False, antialias=True) for e in ee]).clamp(0, 1)
        res = vs.native_residual(cv, bits) - vs.native_residual(cv, 1 - bits)
        for e, rr in zip(ee, res):
            S = 512 + e; R = F.interpolate(rr[None], size=(S, S), mode="bilinear", align_corners=False, antialias=(meth == "pixelseal"))
            if R.shape[1] == 1: R = R.repeat(1, 3, 1, 1)
            Rb = vs.bp(R)
            if e == 0:
                num = (xw * Rb).sum().view(1, 1); den = (w2 * Rb * Rb).sum().view(1, 1)
            else:
                fa = torch.fft.rfft2(Rb); pad = lambda t: F.pad(t, (0, e, 0, e))
                num = torch.fft.irfft2(fa * torch.conj(torch.fft.rfft2(pad(xw))), s=(S, S)).sum((0, 1))[:e + 1, :e + 1]
                den = torch.fft.irfft2(torch.fft.rfft2(Rb * Rb) * torch.conj(torch.fft.rfft2(pad(w2))), s=(S, S)).sum((0, 1))[:e + 1, :e + 1]
            sl = num / den.clamp_min(1e-6); k = int(sl.argmax()); t, l = divmod(k, sl.shape[1])
            out.append((float(sl.flatten()[k]), e, t, l))
    return out

def search(x, bits):
    xb = vs.bp(x * 255); hm = wm.model.attenuation.heatmaps(x) if (meth == "pixelseal" and wm.model.attenuation is not None) else None
    cz = coarse(x, xb, bits, hm); best = {}
    def run(cands):
        cands = [c for c in set(cands) if c not in best and min(c) >= 0 and c[0] + c[1] <= EMAX + 16 and c[2] + c[3] <= EMAX + 16]
        if cands:
            for c, s in zip(cands, vs.scores4(x, xb, bits, cands)): best[c] = s
    sq = lambda e, t, l: (t, e - t, l, e - l)
    run([(0, 0, 0, 0)] + [sq(e, t, l) for _, e, t, l in sorted(cz, reverse=True)[:3]])
    for step in (4, 2, 1):
        a, b, l, r = max(best, key=best.get); e = max(a + b, l + r)
        run([sq(e + d1 * step, a + d2 * step, l + d3 * step) for d1 in (-1, 0, 1) for d2 in (-1, 0, 1) for d3 in (-1, 0, 1)
             if e + d1 * step >= 0 and 0 <= a + d2 * step <= e + d1 * step and 0 <= l + d3 * step <= e + d1 * step])
    a, b, l, r = max(best, key=best.get)
    run([(a + s1, b + s2, l + s3, r + s4) for s1 in (-1, 0, 1) for s2 in (-1, 0, 1) for s3 in (-1, 0, 1) for s4 in (-1, 0, 1)])
    k = max(best, key=best.get)
    return k, best[k], best[(0, 0, 0, 0)], [round(c[0], 3) for c in cz]

wh = np.load(f"{OUT}/whole_{split}.npz"); acc = {str(n): float(v) for n, v in zip(wh["names"], wh["acc"])}
isfull = lambda it: (it["full"] if split == "train" else acc[it["name"]] > 0.70)
items = [it for it in load_items(split, meth) if WHICH == "all" or (isfull(it) == (WHICH == "full"))]
if limit: items = items[:limit]
items = items[SK::SN]; res = {}; t0 = time.time(); tag = "" if WHICH == "full" else f"_{WHICH}"
for j, it in enumerate(items):
    x = load_img(it["path"]); bits = torch.from_numpy(it["bits"].astype(np.float32)).to(dev)
    k, s, s0, curve = search(x, bits)
    res[it["name"]] = dict(a=int(k[0]), b=int(k[1]), l=int(k[2]), r=int(k[3]), slope=float(s), slope_plain=float(s0), e_curve=curve)
    print(f"{j:3d} {it['name']} top {k[0]:3d} bottom {k[1]:3d} left {k[2]:3d} right {k[3]:3d} | extra rows {k[0] + k[1]:3d} cols {k[2] + k[3]:3d} | slope {s:+.3f} (visible frame {s0:+.3f}) | {time.time() - t0:.0f}s", flush=True)
    if j % 10 == 9 or j == len(items) - 1: json.dump(res, open(f"{OUT}/frame{tag}_{meth}_{split}{SUF}.json", "w"))
A = np.array([[r["a"] + r["b"], r["l"] + r["r"], r["slope"], r["slope_plain"]] for r in res.values()])
print(f"== {meth} {split} {WHICH} n={len(A)}: slope > 0.2 on the visible frame {(A[:, 3] > 0.2).mean():.2f} -> after the frame search {(A[:, 2] > 0.2).mean():.2f} | mean slope {A[:, 3].mean():.3f} -> {A[:, 2].mean():.3f}")
print("   extra rows quantiles", np.quantile(A[:, 0], [0, .25, .5, .75, 1]).astype(int), "extra cols", np.quantile(A[:, 1], [0, .25, .5, .75, 1]).astype(int), "| mean |rows - cols|", round(float(np.abs(A[:, 0] - A[:, 1]).mean()), 1))
