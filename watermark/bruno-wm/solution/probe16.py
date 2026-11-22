"""Frame margin of full-watermark challenge images, measured with TrustMark / PixelSeal (fine patterns):
template = embed(reflect-padded image by k px per side, msg) - embed(same, 1-msg), cropped back; correlation with the
band-passed image inside the watermarked region, as a function of k. Control: our own embedding with k_true = 8."""
import sys, numpy as np, torch
from common import RELEASE
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from bands import blur
def bp(a):
    x = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].cuda(); return (blur(x, 0.7) - blur(x, 4.0))[0].mean(0).cpu().numpy()
def tmpl(wm, x, msg, t, b, l, r):
    p = np.pad(x, ((t, b), (l, r), (0, 0)), mode="reflect")
    T = wm.embed(p, msg).astype(np.float32) - wm.embed(p, 1 - msg).astype(np.float32)
    return T[t:t + 512, l:l + 512]
def corr(a, b, M):
    a = a[M] - a[M].mean(); b = b[M] - b[M].mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
KS = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]
for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
    reals = [ds[i] for i in range(len(ds)) if ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() > 0.85][:30]
    cleans = [ds[i] for i in range(len(ds)) if not ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() < 0.05][:12]
    R = np.zeros((len(reals), len(KS)))
    for i, it in enumerate(reals):
        bx = bp(it["image"]); M = it["mask"].astype(bool)
        for j, k in enumerate(KS): R[i, j] = corr(bx, bp(tmpl(wm, it["image"], it["message"], k, k, k, k)), M)
    C = np.zeros((len(cleans), len(KS))); full = np.ones((512, 512), bool)
    for i, it in enumerate(cleans):
        msg = rng.integers(0, 2, wm.nbits); p = np.pad(it["image"], ((8, 8), (8, 8), (0, 0)), mode="reflect"); xs = wm.embed(p, msg)[8:520, 8:520]; bx = bp(xs)
        for j, k in enumerate(KS): C[i, j] = corr(bx, bp(tmpl(wm, xs, msg, k, k, k, k)), full)
    print(f"== {meth}: symmetric margin k:      " + " ".join("%6d" % k for k in KS))
    print("   REAL mean corr:              " + " ".join("%+.3f" % v for v in R.mean(0)))
    print("   CONTROL (true k=8) mean corr:" + " ".join("%+.3f" % v for v in C.mean(0)))
    print("   REAL best k per image: " + " ".join(str(KS[j]) for j in R.argmax(1)), flush=True)
