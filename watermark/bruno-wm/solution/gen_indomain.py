"""In-domain synthetic data for one method: take challenge images that are almost entirely clean with respect to
this method (images of the OTHER two methods whose watermark only covers a small rectangle) and watermark them
ourselves with this method. Their own small rectangle carries a different watermark, so labelling it 'clean' is
correct for this method's detector. Validation images of this method are never used."""
import json, sys, time, numpy as np
from PIL import Image
from common import SYN, RELEASE, OUT, MAX_ASPECT
from gen_wm import sample_rect
from wmloc.watermarks import load_watermark

KF, KR = 3, 6
if __name__ == "__main__":
    meth = sys.argv[1]
    acc = json.load(open(f"{OUT}/global_acc.json"))
    bases = [("train", m["image_name"]) for m in json.load(open(f"{RELEASE}/train/metadata.json")) if m["method"] != meth and not m["full_watermark"]]
    bases += [("test", m["image_name"]) for m in json.load(open(f"{RELEASE}/test/metadata.json")) if m["method"] != meth and acc[m["image_name"]]["acc"] < 0.62]
    N = len(bases); wm = load_watermark(meth, device="cuda"); rng = np.random.default_rng(abs(hash(meth)) % 2**31)
    base = np.lib.format.open_memmap(f"{SYN}/{meth}_in_base.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3))
    full = np.lib.format.open_memmap(f"{SYN}/{meth}_in_full.npy", mode="w+", dtype=np.uint8, shape=(N, KF, 512, 512, 3))
    rect = np.lib.format.open_memmap(f"{SYN}/{meth}_in_rect.npy", mode="w+", dtype=np.uint8, shape=(N, KR, 512, 512, 3))
    rects = np.zeros((N, KR, 4), np.int32); t = time.time()
    for i, (split, name) in enumerate(bases):
        c = np.asarray(Image.open(f"{RELEASE}/{split}/images/{name}.png").convert("RGB"), dtype=np.uint8); base[i] = c
        for k in range(KF):
            full[i, k] = wm.embed(c, rng.integers(0, 2, wm.nbits))
        for k in range(KR):
            r = c.copy()
            for _ in range(5):
                y0, y1, x0, x1 = sample_rect(rng, MAX_ASPECT[meth])
                try:
                    w = wm.embed(np.ascontiguousarray(c[y0:y1, x0:x1]), rng.integers(0, 2, wm.nbits))
                    assert w.shape == (y1 - y0, x1 - x0, 3)
                    r[y0:y1, x0:x1] = w; rects[i, k] = (y0, y1, x0, x1); break
                except Exception:
                    pass
            rect[i, k] = r
    base.flush(); full.flush(); rect.flush(); np.save(f"{SYN}/{meth}_in_rects.npy", rects)
    print("INDOMAIN_DONE", meth, N, f"{time.time()-t:.0f}s")
