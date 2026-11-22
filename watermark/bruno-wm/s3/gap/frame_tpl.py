"""Does the kit template match the REAL residual once the embedding frame is allowed to be larger than the visible image?
For a canvas scale s (canvas = image reflect-padded to 512*s, centred) we embed the known message m and another message
m' with the kit, take T = embed(canvas, m) - embed(canvas, m') cropped back to the visible 512 x 512, and regress the
band-passed image on the band-passed T inside the watermarked area (ground truth). Slope ~0.3-0.5 = the template is the
real pattern; ~0 = no match. Control: the same image re-watermarked by us at canvas scale 1.06 (so the right answer
is known).  usage: frame_tpl.py <method> [n]"""
import sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import *
meth = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
wm = load_wm(meth); dev = "cuda"; rng = np.random.default_rng(0)
SC = [0.94, 0.97, 1.0, 1.01, 1.02, 1.03, 1.04, 1.06, 1.08, 1.10, 1.125, 1.15, 1.20, 1.25]
BANDS = [(1.5, 3), (3, 6), (6, 12)]

def blur(x, s):
    k = int(4 * s) | 1; ax = torch.arange(k, device=x.device) - k // 2; g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum(); C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
def bp(x, lo, hi): return blur(x, lo) - blur(x, hi)
def t(a): return torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].to(dev)

def template(img_u8, bits, s):
    """kit template on a centred canvas of scale s, cropped (s>=1) or padded back (s<1) to 512."""
    H = img_u8.shape[0]; P = int(round(H * (s - 1) / 2))
    if P > 0: c = np.pad(img_u8, ((P, P), (P, P), (0, 0)), mode="reflect")
    elif P < 0: c = img_u8[-P:H + P, -P:H + P]
    else: c = img_u8
    c = np.ascontiguousarray(c); other = 1 - bits
    T = wm.embed(c, bits).astype(np.float32) - wm.embed(c, other).astype(np.float32)
    if P > 0: T = T[P:P + H, P:P + H]
    elif P < 0: T = np.pad(T, ((-P, -P), (-P, -P), (0, 0)))
    return t(T)

def slopes(x, T, m):
    out = []
    for lo, hi in BANDS:
        a = bp(x, lo, hi) * m; b = bp(T, lo, hi) * m; out.append(float((a * b).sum() / (b * b).sum().clamp_min(1e-6)))
    return out

items = [it for it in load_items("train", meth) if it["full"]][:n]
R, Cc = [], []
for it in items:
    u8 = np.asarray(Image.open(it["path"]).convert("RGB"), dtype=np.uint8); x = t(u8); m = torch.from_numpy(gt512(it["name"])).to(dev)[None, None].float()
    R.append([slopes(x, template(u8, it["bits"], s), m) for s in SC])
    # control: re-watermark the same image ourselves on a canvas of scale 1.06 with a NEW message, then search the scale
    nb = rng.integers(0, 2, len(it["bits"])); P = int(round(512 * 0.03)); c = np.ascontiguousarray(np.pad(u8, ((P, P), (P, P), (0, 0)), mode="reflect"))
    cw = wm.embed(c, nb)[P:P + 512, P:P + 512]; xc = t(cw)
    Cc.append([slopes(xc, template(np.ascontiguousarray(cw), nb, s), torch.ones_like(m)) for s in SC])
R = np.array(R); Cc = np.array(Cc)
print(f"== {meth} n={len(items)}  slope of image on kit template, by canvas scale (columns) and band (rows)")
print("   scale      " + " ".join(f"{s:6.3f}" for s in SC))
for b, (lo, hi) in enumerate(BANDS):
    print(f"   REAL  {lo}-{hi:<3} " + " ".join(f"{v:+6.3f}" for v in R[:, :, b].mean(0)))
for b, (lo, hi) in enumerate(BANDS):
    print(f"   CTRL  {lo}-{hi:<3} " + " ".join(f"{v:+6.3f}" for v in Cc[:, :, b].mean(0)) + "   (true scale 1.06)")
b = 1; best = R[:, :, b].argmax(1)
print("   REAL per image, band 3-6: best scale -> slope there | slope at 1.00")
print("   " + "  ".join(f"{SC[k]:.3f}:{R[i, k, b]:+.2f}|{R[i, 2, b]:+.2f}" for i, k in enumerate(best)))
print("   best-scale histogram: " + " ".join(f"{SC[k]}:{(best == k).sum()}" for k in range(len(SC)) if (best == k).sum()))
