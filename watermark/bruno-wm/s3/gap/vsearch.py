"""Estimate, for each whole-frame image, the canvas on which the watermark was embedded, assuming the visible image is
a vertical crop of a taller original: a rows hidden above, b rows hidden below (canvas = 512 x (512 + a + b)).
Objective: regression slope of band-pass(image) on band-pass(kit template computed on the reflect-padded canvas).
Coarse-to-fine search over (a, b). Batched templates on the GPU (same maths as the kit embedders).
usage: vsearch.py <method> <split> [limit]   -> out/canvas_<method>_<split>.json"""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
from common import *
meth, split = sys.argv[1], sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SH = sys.argv[4] if len(sys.argv) > 4 else "0/1"; SK, SN = [int(v) for v in SH.split("/")]; SUF = "" if SN == 1 else f".{SK}of{SN}"
wm = load_wm(meth); dev = "cuda"
LO, HI = (3.0, 6.0) if meth != "pixelseal" else (1.0, 3.0)
EMAX = 512

def blur(x, s):
    k = int(4 * s) | 1; ax = torch.arange(k, device=x.device) - k // 2; g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum(); C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
def bp(x): return blur(x, LO) - blur(x, HI)

def canvas(x, a, b):
    """x: 1 x 3 x 512 x 512 -> 1 x 3 x (512+a+b) x 512, missing rows filled by (repeated) reflection."""
    i = np.arange(-a, 512 + b); p = 2 * 511; i = np.mod(i, p); i = np.where(i > 511, p - i, i)
    return x[:, :, torch.from_numpy(i).to(x.device), :]

@torch.no_grad()
def native_residual(cv, bits):
    """residual at the embedder's native resolution for a batch of canvases already resized to native size, [0,1]."""
    if meth == "trustmark":
        tm = wm.tm; cover = cv * 2 - 1
        out = torch.cat([tm.encoder(cover[i:i + 1], bits)[0] for i in range(len(cover))]).clamp(-1, 1)
        return (out - cover) * 127.5
    if meth == "stegastamp":
        r = wm._run(cv.permute(0, 2, 3, 1).cpu().numpy(), ["residual"], np.repeat(bits.cpu().numpy()[None].astype(np.float32), len(cv), 0))[0]
        return torch.from_numpy(r).to(dev).permute(0, 3, 1, 2) * 255
    raise NotImplementedError

@torch.no_grad()
def scores(x, xb, bits, cands, bs=48):
    """slope for each (a, b)."""
    nat = wm.tm.model_resolution_enc if meth == "trustmark" else 400
    mode = "bilinear" if meth == "trustmark" else "bicubic"
    out = []
    for i in range(0, len(cands), bs):
        cs = cands[i:i + bs]
        cv = torch.cat([F.interpolate(canvas(x, a, b), size=(nat, nat), mode=mode, align_corners=False, antialias=True) for a, b in cs]).clamp(0, 1)
        res = native_residual(cv, bits) - native_residual(cv, 1 - bits)
        for (a, b), r in zip(cs, res):
            T = F.interpolate(r[None], size=(512 + a + b, 512), mode="bilinear", align_corners=False)[:, :, a:a + 512]
            tb = bp(T); out.append(float((xb * tb).sum() / (tb * tb).sum().clamp_min(1e-6)))
    return np.array(out)

def search(x, bits):
    xb = bp(x * 255); best = {}
    def run(cands):
        cands = [c for c in set(cands) if c not in best and c[0] >= 0 and c[1] >= 0 and c[0] + c[1] <= EMAX]
        if cands:
            for c, s in zip(cands, scores(x, xb, bits, cands)): best[c] = s
    run([(a, E - a) for E in range(0, EMAX + 1, 16) for a in range(0, E + 1, 16)])
    for step in (8, 4, 2, 1):
        top = sorted(best, key=best.get, reverse=True)[:3]
        run([(a + da, b + db) for a, b in top for da in (-step, 0, step) for db in (-step, 0, step)])
        top = sorted(best, key=best.get, reverse=True)[:1]
        run([(a + da, b + db) for a, b in top for da in (-step, 0, step) for db in (-step, 0, step)])
    k = max(best, key=best.get); order = sorted(best, key=best.get, reverse=True)
    second = next((c for c in order if abs(c[0] - k[0]) + abs(c[1] - k[1]) > 12), k)
    return k, best[k], best[(0, 0)], second, best[second], len(best)

wh = np.load(f"{OUT}/whole_{split}.npz"); acc = {str(n): float(v) for n, v in zip(wh["names"], wh["acc"])}
items = [it for it in load_items(split, meth) if (it["full"] if split == "train" else acc[it["name"]] > 0.70)]
if limit: items = items[:limit]
items = items[SK::SN]
res = {}; t0 = time.time()
for j, it in enumerate(items):
    x = load_img(it["path"]); bits = torch.from_numpy(it["bits"].astype(np.float32)).to(dev)
    k, s, s0, k2, s2, n = search(x, bits)
    res[it["name"]] = dict(a=int(k[0]), b=int(k[1]), slope=float(s), slope_plain=float(s0), second=[int(k2[0]), int(k2[1])], slope_second=float(s2))
    print(f"{j:3d} {it['name']} above {k[0]:3d} below {k[1]:3d} (height {512 + k[0] + k[1]}) slope {s:+.3f} | plain {s0:+.3f} | runner-up {k2} {s2:+.3f} | {n} canvases {time.time() - t0:.0f}s", flush=True)
    if j % 10 == 9 or j == len(items) - 1: json.dump(res, open(f"{OUT}/canvas_{meth}_{split}{SUF}.json", "w"))
A = np.array([[r["a"], r["b"], r["slope"], r["slope_plain"]] for r in res.values()])
print(f"== {meth} {split} n={len(A)}: plain frame matches (slope_plain > 0.2): {(A[:, 3] > 0.2).mean():.2f} | canvas found (slope > 0.2): {(A[:, 2] > 0.2).mean():.2f} | mean slope {A[:, 2].mean():.3f}")
E = A[:, 0] + A[:, 1]; ok = A[:, 2] > 0.2
print("   extra rows (a+b) among found:", np.round(np.quantile(E[ok], [0, .1, .25, .5, .75, .9, 1])).astype(int), "| share with no extra rows:", round(float((E[ok] == 0).mean()), 2), "| a/(a+b) quantiles:", np.round(np.quantile((A[ok & (E > 0), 0] / E[ok & (E > 0)]), [.1, .25, .5, .75, .9]), 2))
