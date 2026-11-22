"""Estimate, for each whole-frame image, the canvas on which the watermark was embedded, assuming the visible image is
a vertical crop of a taller original: a rows hidden above, b rows hidden below (canvas = 512 x (512 + a + b)).
Objective: regression slope of band-pass(image) on band-pass(kit template computed on the reflect-padded canvas).
Coarse-to-fine search over (a, b). Batched templates on the GPU (same maths as the kit embedders).
usage: vsearch.py <method> <split> [limit]   -> out/canvas_<method>_<split>.json"""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
from common import *
meth, split = sys.argv[1], sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
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
    if meth == "pixelseal":
        mdl = wm.model; msgs = bits[None].repeat(len(cv), 1)
        inp = mdl.rgb2yuv(cv)[:, 0:1] if mdl.embedder.yuv else cv
        return mdl.embedder(inp, msgs).float() * 255 * mdl.blender.scaling_w
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


def canvas4(x, a, b, l, r):
    def idx(lo, hi):
        i = np.arange(-lo, 512 + hi); p = 2 * 511; i = np.mod(i, p); return torch.from_numpy(np.where(i > 511, p - i, i)).to(x.device)
    return x[:, :, idx(a, b), :][:, :, :, idx(l, r)]

@torch.no_grad()
def scores4(x, xb, bits, cands, bs=48):
    nat = {"trustmark": getattr(getattr(wm, "tm", None), "model_resolution_enc", 256), "stegastamp": 400, "pixelseal": 256}[meth]
    mode = "bicubic" if meth == "stegastamp" else "bilinear"
    hm = wm.model.attenuation.heatmaps(x) if (meth == "pixelseal" and wm.model.attenuation is not None) else None
    out = []
    for i in range(0, len(cands), bs):
        cs = cands[i:i + bs]
        cv = torch.cat([F.interpolate(canvas4(x, *c), size=(nat, nat), mode=mode, align_corners=False, antialias=True) for c in cs]).clamp(0, 1)
        res = native_residual(cv, bits) - native_residual(cv, 1 - bits)
        for (a, b, l, r), rr in zip(cs, res):
            T = F.interpolate(rr[None], size=(512 + a + b, 512 + l + r), mode="bilinear", align_corners=False, antialias=(meth == "pixelseal"))[:, :, a:a + 512, l:l + 512]
            if hm is not None: T = T * hm
            if T.shape[1] == 1: T = T.repeat(1, 3, 1, 1)
            tb = bp(T); out.append(float((xb * tb).sum() / (tb * tb).sum().clamp_min(1e-6)))
    return np.array(out)
