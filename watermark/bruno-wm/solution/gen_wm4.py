"""Full-type synthetic watermarks, square-canvas model measured on the challenge images: the embedder ran on a SQUARE
of side 512 + e that contains the visible 512x512 image at a free offset (crop, no resampling).
e per method: StegaStamp ~14-29, TrustMark ~31-80, PixelSeal ~61-128+ (we draw from a wider range)."""
import sys, time, numpy as np
from common import SYN
from wmloc.watermarks import load_watermark
P = 32
E_RANGE = {"stegastamp": (8, 40), "trustmark": (16, 110), "pixelseal": (32, 200)}
def full_variant(wm, canvas, rng, meth):
    lo, hi = E_RANGE[meth]; e = int(rng.integers(lo, hi + 1)); oy, ox = int(rng.integers(0, e + 1)), int(rng.integers(0, e + 1))
    top, left = P - oy, P - ox; bot, right = top + 512 + e, left + 512 + e
    pt, pl, pb, pr = max(0, -top), max(0, -left), max(0, bot - canvas.shape[0]), max(0, right - canvas.shape[1])
    c = np.pad(canvas, ((pt, pb), (pl, pr), (0, 0)), mode="reflect") if (pt or pl or pb or pr) else canvas
    w = wm.embed(np.ascontiguousarray(c[top + pt:bot + pt, left + pl:right + pl]), rng.integers(0, 2, wm.nbits))
    return np.ascontiguousarray(w[oy:oy + 512, ox:ox + 512]), (e, oy, ox)
if __name__ == "__main__":
    meth = sys.argv[1]; NEXT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    TAG, INTAG = "d", "in4"
    if len(sys.argv) > 5:                                   # custom range and tag: gen_wm4.py <method> <n> <e_lo> <e_hi> <tag> <intag>
        E_RANGE[meth] = (int(sys.argv[3]), int(sys.argv[4])); TAG, INTAG = sys.argv[5], sys.argv[6]
    wm = load_watermark(meth, device="cuda"); rng = np.random.default_rng(abs(hash((TAG, meth))) % 2**31); t0 = time.time()
    base = np.load(f"{SYN}/{meth}_in2_base.npy", mmap_mode="r"); N, KF = len(base), 4
    full = np.lib.format.open_memmap(f"{SYN}/{meth}_{INTAG}_full.npy", mode="w+", dtype=np.uint8, shape=(N, KF, 512, 512, 3))
    for i in range(N):
        can = np.pad(np.array(base[i]), ((P, P), (P, P), (0, 0)), mode="reflect")
        for k in range(KF): full[i, k] = full_variant(wm, can, rng, meth)[0]
    full.flush(); print("WM4_INDOMAIN_DONE", meth, N, f"{time.time()-t0:.0f}s", flush=True)
    can = np.load(f"{SYN}/clean576.npy", mmap_mode="r"); N = min(len(can), NEXT); info = np.zeros((N, 3), np.int32)
    full = np.lib.format.open_memmap(f"{SYN}/{meth}_full_{TAG}.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3))
    for i in range(N):
        full[i], info[i] = full_variant(wm, np.array(can[i]), rng, meth)
    full.flush(); np.save(f"{SYN}/{meth}_info_{TAG}.npy", info); print("WM4_DONE", meth, N, f"{time.time()-t0:.0f}s")
