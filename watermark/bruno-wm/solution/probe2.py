"""Matched-filter probe, content-free: T = embed(x, msg) - embed(x, other_msg) is the message-specific part of the
residual our embedder would add. If the image contains our embedder's residual for msg, band-passed x correlates
with band-passed T (expected slope ~0.5); in clean areas the correlation is ~0. Compared per frequency band between
the challenge images (REAL) and images we watermark ourselves (CONTROL)."""
import sys, numpy as np, torch, torch.nn.functional as F
from common import RELEASE
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

def blur(x, s):
    if s <= 0: return x
    r = int(3 * s + 0.5); t = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device); k = torch.exp(-t ** 2 / (2 * s * s)); k /= k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)

SIG = [0, 0.7, 1.5, 3, 6, 12]
def bands(a):
    x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda(); g = [blur(x, s) for s in SIG]
    return [(g[i] - g[i + 1])[0].permute(1, 2, 0).cpu().numpy() for i in range(len(SIG) - 1)]

def stat(x, T, region):
    out = []
    for bx, bt in zip(bands(x), bands(T)):
        a, b = bx[region], bt[region]
        out.append((float((a * b).sum() / max((b * b).sum(), 1e-9)), float(np.sqrt((b * b).mean()))))
    return out

for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
    R = {"REAL wm region": [], "REAL clean holes": [], "CONTROL ours": [], "CONTROL clean": []}; nf = nc = 0
    for i in range(len(ds)):
        it = ds[i]; M = it["mask"].astype(bool)
        if it["full_watermark"] and nf < 40:
            x = it["image"]; msg = it["message"]; nf += 1
            T = wm.embed(x, msg).astype(np.float32) - wm.embed(x, 1 - msg).astype(np.float32)
            R["REAL wm region"].append(stat(x.astype(np.float32), T, M))
            if (~M).sum() > 2000: R["REAL clean holes"].append(stat(x.astype(np.float32), T, ~M))
        elif not it["full_watermark"] and nc < 40:
            c = it["image"]; msg = rng.integers(0, 2, wm.nbits); xs = wm.embed(c, msg); nc += 1
            T = wm.embed(xs, msg).astype(np.float32) - wm.embed(xs, 1 - msg).astype(np.float32)
            R["CONTROL ours"].append(stat(xs.astype(np.float32), T, ~M))
            T0 = wm.embed(c, msg).astype(np.float32) - wm.embed(c, 1 - msg).astype(np.float32)
            R["CONTROL clean"].append(stat(c.astype(np.float32), T0, ~M))
    print(f"== {meth}: slope per band (sigma {SIG[0]}-{SIG[1]}, {SIG[1]}-{SIG[2]}, {SIG[2]}-{SIG[3]}, {SIG[3]}-{SIG[4]}, {SIG[4]}-{SIG[5]}); template RMS per band: "
          + " ".join("%.2f" % v for v in np.mean(np.array(R["CONTROL ours"])[:, :, 1], 0)))
    for k, v in R.items():
        print(f"   {k:17s} " + "  ".join("%+.3f" % s for s in np.mean(np.array(v)[:, :, 0], 0)), flush=True)
