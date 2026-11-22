"""Full-type synthetic watermarks with the canvas model measured on the challenge images (TrustMark / PixelSeal):
the visible 512x512 image is a crop of a larger watermarked canvas = image + ~4 px on two opposite sides and a large
asymmetric extension on the other axis (hidden rows above / below, total 0..~512, more for PixelSeal).
Content outside the image: real pixels where the 576-px pool has them, reflection beyond."""
import json, sys, time, numpy as np
from PIL import Image
from common import SYN, RELEASE, OUT
from wmloc.watermarks import load_watermark
P = 32
def sample_T(rng, meth):
    u = rng.random()
    if meth == "stegastamp": return 8 if u < 0.8 else int(rng.integers(0, 33))        # frame = image + 4 px per side
    if meth == "pixelseal" and u > 0.85: return int(rng.integers(512, 2500))
    return int(rng.integers(0, 81)) if u < 0.5 else int(rng.integers(80, 513))
def full_variant(wm, canvas, rng, meth):
    """canvas: 576x576x3 (image = central 512). Returns the watermarked 512 image and (a, b, vertical)."""
    vert = rng.random() < 0.8
    c = canvas if vert else canvas.transpose(1, 0, 2)
    T = sample_T(rng, meth); a = int(rng.random() * (T + 1)); b = T - a
    if meth == "stegastamp": a = T // 2; b = T - a
    ml, mr = [int(v) for v in rng.integers(3, 6, 2)]
    pa, pb = max(0, a - P), max(0, b - P)
    while pa > 0 or pb > 0:                                   # reflect in steps (np.pad reflect needs pad < size)
        s0, s1 = min(pa, c.shape[0] - 1), min(pb, c.shape[0] - 1); c = np.pad(c, ((s0, s1), (0, 0), (0, 0)), mode="reflect"); pa -= s0; pb -= s1
    top = (c.shape[0] - 512) // 2 if False else None
    off = max(0, a - P) + P                                   # row of the image's first row inside c
    frame = np.ascontiguousarray(c[off - a:off + 512 + b, P - ml:P + 512 + mr])
    if not vert: frame = np.ascontiguousarray(frame.transpose(1, 0, 2))
    w = wm.embed(frame, rng.integers(0, 2, wm.nbits))
    if not vert: w = w.transpose(1, 0, 2)
    out = w[a:a + 512, ml:ml + 512]
    return np.ascontiguousarray(out if vert else out.transpose(1, 0, 2)), (a, b, int(vert))
if __name__ == "__main__":
    meth = sys.argv[1]; wm = load_watermark(meth, device="cuda"); rng = np.random.default_rng(abs(hash(("c", meth))) % 2**31); t0 = time.time()
    base = np.load(f"{SYN}/{meth}_in2_base.npy", mmap_mode="r"); N, KF = len(base), 4
    full = np.lib.format.open_memmap(f"{SYN}/{meth}_in3_full.npy", mode="w+", dtype=np.uint8, shape=(N, KF, 512, 512, 3))
    for i in range(N):
        can = np.pad(np.array(base[i]), ((P, P), (P, P), (0, 0)), mode="reflect")
        for k in range(KF): full[i, k] = full_variant(wm, can, rng, meth)[0]
    full.flush(); print("WM3_INDOMAIN_DONE", meth, N, f"{time.time()-t0:.0f}s", flush=True)
    can = np.load(f"{SYN}/clean576.npy", mmap_mode="r"); N = len(can); info = np.zeros((N, 3), np.int32)
    full = np.lib.format.open_memmap(f"{SYN}/{meth}_full_c.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3))
    for i in range(N):
        full[i], info[i] = full_variant(wm, np.array(can[i]), rng, meth)
        if i % 2000 == 0: print(meth, i, N, f"{time.time()-t0:.0f}s", flush=True)
    full.flush(); np.save(f"{SYN}/{meth}_info_c.npy", info); print("WM3_DONE", meth, N, f"{time.time()-t0:.0f}s")
