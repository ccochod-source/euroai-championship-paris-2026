"""Hypothesis: the real 'full' images are 512 x 512 CROPS of a larger watermarked original (wider or taller).
For each real full image, search the canvas (orientation, size ratio r = long side / 512, crop position) whose kit
template best matches the image: slope of band-passed image on band-passed template (band 3-6 px), inside the
watermarked area. Unknown margins are filled by reflection.  usage: frame_tpl2.py <method> [n] [step_ratio]"""
import sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import *
meth = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
wm = load_wm(meth); dev = "cuda"
RAT = [1 + k / 16 for k in range(0, 17)]

def blur(x, s):
    k = int(4 * s) | 1; ax = torch.arange(k, device=x.device) - k // 2; g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum(); C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
def t(a): return torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].to(dev)

def template(u8, bits, py0, py1, px0, px1):
    c = np.ascontiguousarray(np.pad(u8, ((py0, py1), (px0, px1), (0, 0)), mode="reflect"))
    T = wm.embed(c, bits).astype(np.float32) - wm.embed(c, 1 - bits).astype(np.float32)
    return t(T[py0:py0 + 512, px0:px0 + 512])

def slope(xb, T, m):
    b = (blur(T, 3) - blur(T, 6)) * m
    return float((xb * b).sum() / (b * b).sum().clamp_min(1e-6))

items = [it for it in load_items("train", meth) if it["full"]][:n]
print(f"== {meth}: best canvas per real full image (orientation, long side / 512, crop position 0=start 0.5=centre 1=end) -> slope | slope on the plain 512 frame")
res = []
for it in items:
    u8 = np.asarray(Image.open(it["path"]).convert("RGB"), dtype=np.uint8); x = t(u8); m = torch.from_numpy(gt512(it["name"])).to(dev)[None, None].float()
    xb = (blur(x, 3) - blur(x, 6)) * m
    cand = []
    for r in RAT:
        E = int(round(512 * (r - 1)))
        for pos in ([0.5] if E == 0 else [0.0, 0.25, 0.5, 0.75, 1.0]):
            a = int(round(E * pos)); b = E - a
            cand.append(("wide", r, pos, slope(xb, template(u8, it["bits"], 0, 0, a, b), m)))
            if E: cand.append(("tall", r, pos, slope(xb, template(u8, it["bits"], a, b, 0, 0), m)))
    s0 = cand[0][3]; best = max(cand, key=lambda c: c[3]); res.append((best, s0))
    top = sorted(cand, key=lambda c: -c[3])[:3]
    print(f"   {it['name']} plain {s0:+.2f} | best " + "  ".join(f"{o} r={r:.4f} pos={p:.2f}: {s:+.2f}" for o, r, p, s in top), flush=True)
B = np.array([b[0][3] for b in res]); S0 = np.array([b[1] for b in res])
print(f"   share matching on the plain frame (slope > 0.2): {(S0 > 0.2).mean():.2f} | after the canvas search: {(B > 0.2).mean():.2f} | mean best slope {B.mean():.3f}")
