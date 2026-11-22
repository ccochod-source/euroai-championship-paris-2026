"""Build the pool of clean 512x512 images (uint8 memmap) from FLUX-1024 (downscaled) and COCO val2017."""
import glob, os, sys, numpy as np
from multiprocessing import Pool
from PIL import Image
from common import CLEAN, SYN

def load(p):
    try:
        im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
        if s < 400: return None
        l, t = (w - s) // 2, (h - s) // 2
        im = im.crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    files = sorted(glob.glob(f"{CLEAN}/flux1024/**/*.jpg", recursive=True)) + sorted(glob.glob(f"{CLEAN}/val2017/*.jpg"))
    rng = np.random.default_rng(0); rng.shuffle(files)
    if limit: files = files[:limit]
    os.makedirs(SYN, exist_ok=True)
    tmp = np.lib.format.open_memmap(f"{SYN}/clean_tmp.npy", mode="w+", dtype=np.uint8, shape=(len(files), 512, 512, 3))
    n = 0
    with Pool(30) as pool:
        for a in pool.imap(load, files, chunksize=16):
            if a is not None:
                tmp[n] = a; n += 1
    tmp.flush()
    out = np.lib.format.open_memmap(f"{SYN}/clean.npy", mode="w+", dtype=np.uint8, shape=(n, 512, 512, 3))
    out[:] = tmp[:n]; out.flush(); del tmp; os.remove(f"{SYN}/clean_tmp.npy")
    print("CLEAN_DONE", n, "of", len(files))
