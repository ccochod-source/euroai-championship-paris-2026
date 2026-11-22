"""Per-image shift search for TrustMark on full-watermark challenge images: cross-correlate the band-passed image
with the template of the 512 frame over shifts within +-40 px; report each image's peak z-score and shift.
Control: our embedding on a frame with random asymmetric margins (true shift = (t-b)/2-ish, (l-r)/2-ish)."""
import sys, numpy as np, torch
from common import RELEASE
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from bands import blur
meth = sys.argv[1]; S = 40
def bp(a):
    x = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].cuda(); return (blur(x, 1.0) - blur(x, 5.0))[0].mean(0)
def peak(x, T, M=None):
    a = bp(x); b = bp(T)
    if M is not None: mm = torch.from_numpy(M).cuda(); a = a * mm; b = b * mm
    a = a - a.mean(); b = b - b.mean()
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(b)), s=(512, 512)); c = torch.fft.fftshift(c)[256 - S:256 + S + 1, 256 - S:256 + S + 1]
    z = (c - c.mean()) / c.std(); k = int(z.argmax()); return float(z.max()), k // (2 * S + 1) - S, k % (2 * S + 1) - S, float(z[S, S])
wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
reals = [ds[i] for i in range(len(ds)) if ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() > 0.85][:24]
cleans = [ds[i] for i in range(len(ds)) if not ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() < 0.05][:10]
print("CONTROL: margins (t,b,l,r) -> peak z at (dy,dx), z at zero shift")
for it in cleans:
    msg = rng.integers(0, 2, wm.nbits); t, b, l, r = [int(v) for v in rng.integers(0, 21, 4)]
    p = np.pad(it["image"], ((t, b), (l, r), (0, 0)), mode="reflect"); xs = wm.embed(p, msg)[t:t + 512, l:l + 512]
    T = wm.embed(xs, msg).astype(np.float32) - wm.embed(xs, 1 - msg).astype(np.float32)
    z, dy, dx, z0 = peak(xs, T); print(f"   margins ({t},{b},{l},{r}) -> z {z:5.1f} at ({dy},{dx}), z0 {z0:5.1f}")
print("REAL full-watermark images: peak z at (dy,dx), z at zero shift")
out = []
for it in reals:
    T = wm.embed(it["image"], it["message"]).astype(np.float32) - wm.embed(it["image"], 1 - it["message"]).astype(np.float32)
    z, dy, dx, z0 = peak(it["image"], T, it["mask"].astype(np.float32)); out.append((z, dy, dx, z0))
print("   " + "  ".join(f"z{z:.1f}@({dy},{dx})/z0={z0:.1f}" for z, dy, dx, z0 in out))
