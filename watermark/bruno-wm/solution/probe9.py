"""On what canvas was the challenge watermark embedded? Build the message-specific pattern at the embedder's native
resolution (averaged over several covers), resize it to a range of canvas sizes and cross-correlate with the challenge
image over all shifts. A sharp peak at canvas = 512 px / shift 0 means 'embedded on the whole 512 image'."""
import sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import RELEASE
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from bands import blur
NATIVE = {"trustmark": 256, "pixelseal": 256, "stegastamp": 400}
def bp(t):   # t: 1x3xHxW cuda float -> HxW band-passed grey
    return (blur(t, 0.7) - blur(t, 3.0))[0].mean(0)
def tt(a): return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32)).permute(2, 0, 1)[None].cuda()
for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0); nat = NATIVE[meth]
    covers = [np.asarray(Image.fromarray(ds[i]["image"]).resize((nat, nat), Image.Resampling.LANCZOS)) for i in range(len(ds)) if not ds.meta[i]["full_watermark"]][:8]
    def pattern(msg):
        D = [wm.embed(c, msg).astype(np.float32) - wm.embed(c, 1 - msg).astype(np.float32) for c in covers]
        B = torch.stack([bp(tt(d)).flatten() for d in D]); B = B / B.norm(dim=1, keepdim=True)
        cc = (B @ B.T); n = len(D); indep = float((cc.sum() - n) / (n * n - n))
        return tt(np.mean(D, 0)), indep
    sizes = [256, 320, 384, 448, 480, 500, 512, 524, 544, 576, 640, 768, 1024]
    res = {s: [] for s in sizes}; ind = []; ctrl = {s: [] for s in sizes}
    reals = [ds[i] for i in range(len(ds)) if ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() > 0.9][:12]
    ours = [ds[i] for i in range(len(ds)) if not ds.meta[i]["full_watermark"]][8:14]
    def scan(img, U, store):
        a = bp(tt(img)); a = a - a.mean()
        for s in sizes:
            u = F.interpolate(U, size=(s, s), mode="bicubic", align_corners=False, antialias=True); b = bp(u); b = b - b.mean()
            P = 512 + s; A = torch.zeros(P, P, device="cuda"); Bm = torch.zeros(P, P, device="cuda"); A[:512, :512] = a; Bm[:s, :s] = b
            c = torch.fft.irfft2(torch.fft.rfft2(A) * torch.conj(torch.fft.rfft2(Bm)), s=(P, P))
            z = float((c.max() - c.mean()) / c.std()); k = int(c.argmax()); dy, dx = k // P, k % P
            dy = dy - P if dy > P // 2 else dy; dx = dx - P if dx > P // 2 else dx
            store[s].append((z, dy, dx))
    for it in reals:
        U, i_ = pattern(it["message"]); ind.append(i_); scan(it["image"], U, res)
    for it in ours:
        m = rng.integers(0, 2, wm.nbits); U, _ = pattern(m); scan(wm.embed(it["image"], m), U, ctrl)
    print(f"== {meth}: pattern similarity across different covers (1 = content-independent): {np.mean(ind):.3f}")
    print("   canvas size : " + " ".join("%6d" % s for s in sizes))
    print("   REAL  peak z: " + " ".join("%6.1f" % np.mean([r[0] for r in res[s]]) for s in sizes))
    print("   OURS  peak z: " + " ".join("%6.1f" % np.mean([r[0] for r in ctrl[s]]) for s in sizes))
    best = max(sizes, key=lambda s: np.mean([r[0] for r in res[s]]))
    print(f"   REAL best canvas {best}: shifts (dy,dx) per image: " + " ".join(f"({r[1]},{r[2]})" for r in res[best]), flush=True)
