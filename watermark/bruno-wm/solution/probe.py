"""Is the challenge watermark the same signal as wm.embed() at 512x512? Correlate the image's band-passed content
with the residual our embedder produces for the known message, inside the ground-truth watermarked region and
inside the clean holes. Control: an image we watermarked ourselves."""
import sys, numpy as np, torch, torch.nn.functional as F
from common import RELEASE
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

def blur(a, s):
    r = int(3 * s + 0.5); t = torch.arange(-r, r + 1, dtype=torch.float32); k = torch.exp(-t ** 2 / (2 * s * s)); k /= k.sum()
    x = torch.from_numpy(a).permute(2, 0, 1)[None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
    return x[0].permute(1, 2, 0).numpy()

def stats(x, d, region, s):
    """slope and correlation of bandpass(x) against bandpass(d) over region (bool HxW)."""
    bx = (x - blur(x, s))[region]; bd = (d - blur(d, s))[region]
    if bd.size < 500 or (bd ** 2).sum() < 1e-6: return np.nan, np.nan
    return float((bx * bd).sum() / (bd * bd).sum()), float(np.corrcoef(bx.ravel(), bd.ravel())[0, 1])

for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
    for s in (1.0, 3.0):
        real_in, real_out, ctrl, wrong = [], [], [], []; nf = nc = 0
        for i in range(len(ds)):
            it = ds[i]; x = it["image"].astype(np.float32); M = it["mask"].astype(bool)
            if it["full_watermark"] and nf < 30:
                d = wm.embed(it["image"], it["message"]).astype(np.float32) - x
                real_in.append(stats(x, d, M, s)); real_out.append(stats(x, d, ~M, s))
                d2 = wm.embed(it["image"], rng.integers(0, 2, wm.nbits)).astype(np.float32) - x      # wrong message
                wrong.append(stats(x, d2, M, s)); nf += 1
            elif not it["full_watermark"] and nc < 30:
                b = rng.integers(0, 2, wm.nbits); xs = wm.embed(it["image"], b)
                d = wm.embed(xs, b).astype(np.float32) - xs.astype(np.float32)
                ctrl.append(stats(xs.astype(np.float32), d, ~M, s)); nc += 1
        f = lambda v: "slope %.3f corr %.3f" % tuple(np.nanmean(np.array(v), 0))
        print(f"{meth} sigma={s}: REAL wm region [{f(real_in)}] | REAL holes [{f(real_out)}] | REAL wrong msg [{f(wrong)}] | CONTROL ours [{f(ctrl)}]", flush=True)
