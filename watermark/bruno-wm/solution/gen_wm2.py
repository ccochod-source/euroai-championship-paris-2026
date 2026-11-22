"""Synthetic watermarks with the frame model observed on the challenge images:
  - partial images: the embedder ran on the rectangle expanded by ~4 px per side, only the inner rectangle is kept;
  - full images: the embedder ran on a frame larger than the image by a random margin per side (image = crop).
External pool: 576-px canvases (real content around the 512 image). In-domain pool (--indomain): challenge images
that are clean for this method, reflect-padded by 32 px."""
import json, sys, time, numpy as np
from PIL import Image
from common import SYN, RELEASE, OUT, MAX_ASPECT
from gen_wm import sample_rect
from wmloc.watermarks import load_watermark
P = 32
def make(wm, canvas, rng, meth, kf, kr):
    """canvas: 576x576x3 uint8, image = canvas[32:544, 32:544]. Returns full variants, rect variants, rects."""
    img = canvas[P:P + 512, P:P + 512]; fulls, rects_img, rects = [], [], []
    for _ in range(kf):
        t, b, l, r = [int(v) for v in rng.integers(0, 21, 4)]
        w = wm.embed(np.ascontiguousarray(canvas[P - t:P + 512 + b, P - l:P + 512 + r]), rng.integers(0, 2, wm.nbits))
        fulls.append(w[t:t + 512, l:l + 512])
    for _ in range(kr):
        out = img.copy(); rc = (0, 0, 0, 0)
        for _ in range(5):
            y0, y1, x0, x1 = sample_rect(rng, MAX_ASPECT[meth]); m = int(rng.integers(3, 6))
            try:
                w = wm.embed(np.ascontiguousarray(canvas[P + y0 - m:P + y1 + m, P + x0 - m:P + x1 + m]), rng.integers(0, 2, wm.nbits))
                assert w.shape == (y1 - y0 + 2 * m, x1 - x0 + 2 * m, 3)
                out[y0:y1, x0:x1] = w[m:-m, m:-m]; rc = (y0, y1, x0, x1); break
            except Exception as e:
                pass
        rects_img.append(out); rects.append(rc)
    return fulls, rects_img, rects
if __name__ == "__main__":
    meth = sys.argv[1]; indomain = len(sys.argv) > 2 and sys.argv[2] == "--indomain"
    wm = load_watermark(meth, device="cuda"); rng = np.random.default_rng(abs(hash((meth, indomain))) % 2**31); t0 = time.time()
    if indomain:
        acc = json.load(open(f"{OUT}/global_acc.json"))
        bases = [("train", m["image_name"]) for m in json.load(open(f"{RELEASE}/train/metadata.json")) if m["method"] != meth and not m["full_watermark"]]
        bases += [("test", m["image_name"]) for m in json.load(open(f"{RELEASE}/test/metadata.json")) if m["method"] != meth and acc[m["image_name"]]["acc"] < 0.62]
        N, KF, KR = len(bases), 3, 6
        full = np.lib.format.open_memmap(f"{SYN}/{meth}_in2_full.npy", mode="w+", dtype=np.uint8, shape=(N, KF, 512, 512, 3))
        rect = np.lib.format.open_memmap(f"{SYN}/{meth}_in2_rect.npy", mode="w+", dtype=np.uint8, shape=(N, KR, 512, 512, 3)); R = np.zeros((N, KR, 4), np.int32)
        base = np.lib.format.open_memmap(f"{SYN}/{meth}_in2_base.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3))
        for i, (split, name) in enumerate(bases):
            c = np.asarray(Image.open(f"{RELEASE}/{split}/images/{name}.png").convert("RGB"), dtype=np.uint8); base[i] = c
            f, r, rc = make(wm, np.pad(c, ((P, P), (P, P), (0, 0)), mode="reflect"), rng, meth, KF, KR)
            full[i] = np.stack(f); rect[i] = np.stack(r); R[i] = rc
        np.save(f"{SYN}/{meth}_in2_rects.npy", R); print("WM2_INDOMAIN_DONE", meth, N, f"{time.time()-t0:.0f}s")
    else:
        can = np.load(f"{SYN}/clean576.npy", mmap_mode="r"); N = len(can)
        full = np.lib.format.open_memmap(f"{SYN}/{meth}_full_b.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3))
        rect = np.lib.format.open_memmap(f"{SYN}/{meth}_rect_b.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3)); R = np.zeros((N, 4), np.int32)
        for i in range(N):
            f, r, rc = make(wm, np.array(can[i]), rng, meth, 1, 1); full[i] = f[0]; rect[i] = r[0]; R[i] = rc[0]
            if i % 2000 == 0: print(meth, i, N, f"{time.time()-t0:.0f}s", flush=True)
        np.save(f"{SYN}/{meth}_rects_b.npy", R); print("WM2_DONE", meth, N, f"{time.time()-t0:.0f}s")
