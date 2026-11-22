"""Measure the geometric relation between the residual present in REAL StegaStamp images and the kit template:
normalised cross-correlation (FFT) between the band-passed image and the band-passed template over all shifts, for
several rescalings of the template. Reports peak height and peak position. Control: in-domain synthetic images."""
import os, sys, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
meth = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
wm = load_watermark(meth, device="cuda")

def bp(x, s1=1.0, s2=8.0):
    x = x.astype(np.float32).mean(2); return cv2.GaussianBlur(x, (0, 0), s1) - cv2.GaussianBlur(x, (0, 0), s2)

def xcorr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
    c = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real / a.size
    k = np.unravel_index(np.argmax(c), c.shape); dy, dx = [(v + 256) % 512 - 256 for v in k]
    return float(c.max()), int(dy), int(dx), float(c[0, 0]), float(np.sort(c.ravel())[-20])

def analyse(X, msg):
    T = wm.embed(X, msg).astype(np.float32) - X; A = bp(X); best = None
    for s in (0.94, 0.96, 0.98, 0.99, 1.0, 1.01, 1.02, 1.04, 1.06):
        Ts = cv2.resize(T, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        h = Ts.shape[0]
        if h >= 512: o = (h - 512) // 2; Ts = Ts[o:o + 512, o:o + 512]
        else: o = (512 - h) // 2; Ts = cv2.copyMakeBorder(Ts, o, 512 - h - o, o, 512 - h - o, cv2.BORDER_CONSTANT, value=0)
        r = xcorr(A, bp(Ts)) + (s,)
        if best is None or r[0] > best[0]: best = r
        if s == 1.0: r1 = r
    return r1, best

ds = LocalisationDataset(RELEASE, "train", methods=[meth])
real = [(ds[i]["image"], ds[i]["message"]) for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:N]
Fu = np.load(f"{SYN}/{meth}_in_full.npy", mmap_mode="r"); M = np.load(f"{out}/{meth}_in_msgs.npy")
print(f"{meth}: cross-correlation of image band-pass with template band-pass")
print("   at scale 1.0: peak | peak shift (dy,dx) | value at zero shift | 20th largest (noise floor)   ||  best scale: peak | shift | scale")
for nm, data in (("CONTROL", [(np.array(Fu[k, 0]), M[k, 0]) for k in range(4)]), ("REAL   ", real)):
    for X, msg in data:
        r1, b = analyse(X, msg)
        print("  %s  %.3f | (%+4d,%+4d) | %.3f | %.3f   ||  %.3f | (%+4d,%+4d) | %.2f" % (nm, r1[0], r1[1], r1[2], r1[3], r1[4], b[0], b[1], b[2], b[5]), flush=True)
