"""TrustMark canvas search with averaging over images. The message-specific TrustMark pattern is fairly similar across
covers, so: build it at 256 px from a few covers, resize to a candidate canvas size, cross-correlate with each
challenge image over all shifts, and AVERAGE the normalised correlation surfaces over many images (the canvas
geometry is assumed common). A consistent geometry shows up as a sharp peak in the averaged surface."""
import sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import RELEASE
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from bands import blur
meth = "trustmark"; nat = 256
def bp(t): return (blur(t, 1.0) - blur(t, 5.0))[0].mean(0)
def tt(a): return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32)).permute(2, 0, 1)[None].cuda()
wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
covers = [np.asarray(Image.fromarray(ds[i]["image"]).resize((nat, nat), Image.Resampling.LANCZOS)) for i in range(len(ds)) if not ds.meta[i]["full_watermark"]][:6]
def pattern(msg): return tt(np.mean([wm.embed(c, msg).astype(np.float32) - wm.embed(c, 1 - msg).astype(np.float32) for c in covers], 0))
sizes = [128, 192, 256, 320, 384, 448, 512, 576, 640, 768, 896, 1024]
VAR = {"id": lambda u: u, "hflip": lambda u: u.flip(-1), "vflip": lambda u: u.flip(-2), "rot180": lambda u: u.flip(-1).flip(-2), "transpose": lambda u: u.transpose(-1, -2)}
def scan(items, getimg):
    acc = {}
    for it, msg in items:
        U = pattern(msg); a = bp(tt(getimg(it, msg))); a = (a - a.mean()) / a.std()
        for s in sizes:
            for vn, vf in (VAR.items() if s == 512 else [("id", VAR["id"])]):
                u = F.interpolate(vf(U), size=(s, s), mode="bicubic", align_corners=False, antialias=True); b = bp(u); b = (b - b.mean()) / b.std()
                P = 512 + s; A = torch.zeros(P, P, device="cuda"); Bm = torch.zeros(P, P, device="cuda"); A[:512, :512] = a; Bm[:s, :s] = b
                c = torch.fft.irfft2(torch.fft.rfft2(A) * torch.conj(torch.fft.rfft2(Bm)), s=(P, P)) / (512 * min(s, 512))
                acc[(s, vn)] = acc.get((s, vn), 0) + c / len(items)
    out = []
    for (s, vn), c in acc.items():
        P = c.shape[0]; z = float((c.max() - c.mean()) / c.std()); k = int(c.argmax()); dy, dx = k // P, k % P
        dy = dy - P if dy > P // 2 else dy; dx = dx - P if dx > P // 2 else dx
        out.append((s, vn, z, float(c.max()), dy, dx, float(c[0, 0])))
    return out
reals = [(ds[i], ds[i]["message"]) for i in range(len(ds)) if ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() > 0.85][:60]
ours = [(ds[i], rng.integers(0, 2, wm.nbits)) for i in range(len(ds)) if not ds.meta[i]["full_watermark"]][6:36]
print("CONTROL (our embedding on the whole 512 image): expect a peak at canvas 512, shift (0,0)")
for s, vn, z, mx, dy, dx, c0 in scan(ours, lambda it, m: wm.embed(it["image"], m)): print(f"   canvas {s:5d} {vn:9s} z {z:6.1f} peak {mx:+.4f} at ({dy},{dx})  value at zero shift {c0:+.4f}")
print(f"REAL challenge images (n={len(reals)})", flush=True)
for s, vn, z, mx, dy, dx, c0 in scan(reals, lambda it, m: it["image"]): print(f"   canvas {s:5d} {vn:9s} z {z:6.1f} peak {mx:+.4f} at ({dy},{dx})  value at zero shift {c0:+.4f}", flush=True)
