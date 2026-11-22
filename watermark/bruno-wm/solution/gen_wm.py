"""For one watermark method, pre-compute for every clean image:
  full[i]  = whole image watermarked with a random message
  rect[i]  = clean image with one random rectangle watermarked (embedder run on the crop), rects[i] = (y0, y1, x0, x1)
This mirrors how the challenge data was built (rectangular watermarked subset, then compositing with the clean image)."""
import sys, time, numpy as np, torch
from common import SYN, MAX_ASPECT
from wmloc.watermarks import load_watermark

def sample_rect(rng, max_aspect):
    for _ in range(100):
        area = np.exp(rng.uniform(np.log(0.003), np.log(0.55))) * 512 * 512
        asp = np.exp(rng.uniform(0, np.log(max_aspect)))
        h, w = np.sqrt(area / asp), np.sqrt(area * asp)
        if rng.random() < 0.5: h, w = w, h
        h, w = int(round(min(h, 512))), int(round(min(w, 512)))
        if min(h, w) < 8 or max(h, w) / min(h, w) > max_aspect: continue
        y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1))
        return y0, y0 + h, x0, x0 + w
    return 128, 384, 128, 384

if __name__ == "__main__":
    meth = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "a"
    clean = np.load(f"{SYN}/clean.npy", mmap_mode="r"); N = len(clean)
    wm = load_watermark(meth, device="cuda")
    rng = np.random.default_rng({"pixelseal": 1, "trustmark": 2, "stegastamp": 3}[meth] + 100 * (ord(tag) - 97))
    full = np.lib.format.open_memmap(f"{SYN}/{meth}_full_{tag}.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3))
    rect = np.lib.format.open_memmap(f"{SYN}/{meth}_rect_{tag}.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3))
    rects = np.zeros((N, 4), np.int32); msgs = np.zeros((N, 2, wm.nbits), np.uint8)
    t = time.time()
    for i in range(N):
        c = np.array(clean[i])
        b = rng.integers(0, 2, wm.nbits); msgs[i, 0] = b
        full[i] = wm.embed(c, b)
        for _ in range(5):
            y0, y1, x0, x1 = sample_rect(rng, MAX_ASPECT[meth])
            b2 = rng.integers(0, 2, wm.nbits)
            try:
                w = wm.embed(np.ascontiguousarray(c[y0:y1, x0:x1]), b2)
                assert w.shape == (y1 - y0, x1 - x0, 3)
                break
            except Exception as e:
                if i < 20: print("embed failed", (y0, y1, x0, x1), repr(e)[:200], flush=True)
                w = None
        r = c.copy()
        if w is not None:
            r[y0:y1, x0:x1] = w; rects[i] = (y0, y1, x0, x1); msgs[i, 1] = b2
        rect[i] = r
        if i % 1000 == 0: print(meth, i, N, f"{time.time()-t:.0f}s", flush=True)
    full.flush(); rect.flush()
    np.save(f"{SYN}/{meth}_rects_{tag}.npy", rects); np.save(f"{SYN}/{meth}_msgs_{tag}.npy", msgs)
    print("WM_DONE", meth, N, f"{time.time()-t:.0f}s")
