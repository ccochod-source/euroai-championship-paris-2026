"""Is the challenge watermark shifted / rescaled relative to wm.embed() on the 512x512 image? Cross-correlate the
band-passed image with the message-specific template over shifts, in a 4x4 grid of windows, averaged over images."""
import sys, numpy as np, torch
from common import RELEASE
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from probe2 import blur
R = 12
def bp(a):
    x = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].cuda()
    return (blur(x, 0.7) - blur(x, 3.0))[0].mean(0).cpu().numpy()
def xcorr_grid(a, b):
    """a, b: 512x512. Returns [4,4,2R+1,2R+1] normalised cross-correlation per 128-px window; out[.., R+dy, R+dx] high
    means a(y, x) ~ b(y - dy, x - dx), i.e. the pattern in a is the template shifted by (dy, dx)."""
    out = np.zeros((4, 4, 2 * R + 1, 2 * R + 1), np.float32)
    for i in range(4):
        for j in range(4):
            p = a[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]; q = b[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            p = p - p.mean(); q = q - q.mean()
            c = np.fft.ifft2(np.fft.fft2(p) * np.conj(np.fft.fft2(q))).real / (np.sqrt((p ** 2).sum() * (q ** 2).sum()) + 1e-9)
            c = np.fft.fftshift(c); out[i, j] = c[64 - R:64 + R + 1, 64 - R:64 + R + 1]
    return out
for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
    acc = {"REAL": [], "CONTROL": []}; nf = nc = 0
    for i in range(len(ds)):
        it = ds[i]
        if it["full_watermark"] and nf < 40 and it["mask"].mean() > 0.9:
            x = it["image"]; msg = it["message"]; nf += 1
            T = wm.embed(x, msg).astype(np.float32) - wm.embed(x, 1 - msg).astype(np.float32)
            acc["REAL"].append(xcorr_grid(bp(x), bp(T)))
        elif not it["full_watermark"] and nc < 40:
            msg = rng.integers(0, 2, wm.nbits); xs = wm.embed(it["image"], msg); nc += 1
            T = wm.embed(xs, msg).astype(np.float32) - wm.embed(xs, 1 - msg).astype(np.float32)
            acc["CONTROL"].append(xcorr_grid(bp(xs), bp(T)))
    for k, v in acc.items():
        m = np.mean(v, 0); print(f"== {meth} {k} (n={len(v)}): value at zero shift per window / peak value @ (dy,dx)")
        for i in range(4):
            row = []
            for j in range(4):
                c = m[i, j]; p = np.unravel_index(c.argmax(), c.shape)
                row.append(f"{c[R, R]:+.3f}/{c.max():+.3f}@({p[0]-R:+d},{p[1]-R:+d})")
            print("   " + "  ".join(row), flush=True)
