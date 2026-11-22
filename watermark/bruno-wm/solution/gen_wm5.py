"""Full-type synthetic watermarks, rectangular-canvas model: the visible 512x512 image is a crop (no resampling) of a
larger original, and the embedder ran on a rectangle of that original which contains the window. Hidden margins per
axis: 65 % total 0-80 px, 35 % total 150-512 px (StegaStamp: 8-40 px), split at random between the two sides; half of
the samples use (nearly) the same total on both axes, half independent totals. TrustMark keeps aspect <= 2."""
import sys, time, numpy as np
from common import SYN
from wmloc.watermarks import load_watermark
P = 32
def total(rng, meth):
    if meth == "stegastamp": return int(rng.integers(8, 41))
    return int(rng.integers(0, 81)) if rng.random() < 0.65 else int(rng.integers(150, 513))
def margins(rng, meth):
    for _ in range(50):
        ty = total(rng, meth); tx = int(np.clip(ty + rng.integers(-6, 7), 0, 512)) if rng.random() < 0.5 else total(rng, meth)
        h, w = 512 + ty, 512 + tx
        if meth != "trustmark" or max(h, w) / min(h, w) <= 2.0: break
    a = int(rng.random() * (ty + 1)); l = int(rng.random() * (tx + 1))
    return a, ty - a, l, tx - l
def full_variant(wm, canvas, rng, meth):
    a, b, l, r = margins(rng, meth); top, left = P - a, P - l; bot, right = P + 512 + b, P + 512 + r
    pt, pl, pb, pr = max(0, -top), max(0, -left), max(0, bot - canvas.shape[0]), max(0, right - canvas.shape[1])
    c = np.pad(canvas, ((pt, pb), (pl, pr), (0, 0)), mode="reflect") if (pt or pl or pb or pr) else canvas
    w = wm.embed(np.ascontiguousarray(c[top + pt:bot + pt, left + pl:right + pl]), rng.integers(0, 2, wm.nbits))
    return np.ascontiguousarray(w[a:a + 512, l:l + 512]), (a, b, l, r)
if __name__ == "__main__":
    meth = sys.argv[1]; NEXT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    wm = load_watermark(meth, device="cuda"); rng = np.random.default_rng(abs(hash(("e", meth))) % 2**31); t0 = time.time()
    base = np.load(f"{SYN}/{meth}_in2_base.npy", mmap_mode="r"); N, KF = len(base), 4
    full = np.lib.format.open_memmap(f"{SYN}/{meth}_in5_full.npy", mode="w+", dtype=np.uint8, shape=(N, KF, 512, 512, 3))
    for i in range(N):
        can = np.pad(np.array(base[i]), ((P, P), (P, P), (0, 0)), mode="reflect")
        for k in range(KF): full[i, k] = full_variant(wm, can, rng, meth)[0]
    full.flush(); print("WM5_INDOMAIN_DONE", meth, N, f"{time.time()-t0:.0f}s", flush=True)
    can = np.load(f"{SYN}/clean576.npy", mmap_mode="r"); N = min(len(can), NEXT); info = np.zeros((N, 4), np.int32)
    full = np.lib.format.open_memmap(f"{SYN}/{meth}_full_e.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3))
    for i in range(N):
        full[i], info[i] = full_variant(wm, np.array(can[i]), rng, meth)
    full.flush(); np.save(f"{SYN}/{meth}_info_e.npy", info); print("WM5_DONE", meth, N, f"{time.time()-t0:.0f}s")
