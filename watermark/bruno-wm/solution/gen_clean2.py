"""Clean pool at 576x576: the 512x512 'image' is the central crop, the surrounding 32 px give real content for
watermark frames that extend beyond the image (see gen_wm2.py)."""
import glob, os, sys, numpy as np
from multiprocessing import Pool
from PIL import Image
from common import CLEAN, SYN
S = 576
def load(p):
    try:
        im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
        if s < 440: return None
        l, t = (w - s) // 2, (h - s) // 2
        return np.asarray(im.crop((l, t, l + s, t + s)).resize((S, S), Image.Resampling.LANCZOS), dtype=np.uint8)
    except Exception:
        return None
if __name__ == "__main__":
    files = sorted(glob.glob(f"{CLEAN}/flux1024/**/*.jpg", recursive=True)) + sorted(glob.glob(f"{CLEAN}/val2017/*.jpg"))
    rng = np.random.default_rng(1); rng.shuffle(files)
    tmp = np.lib.format.open_memmap(f"{SYN}/clean576_tmp.npy", mode="w+", dtype=np.uint8, shape=(len(files), S, S, 3)); n = 0
    with Pool(24) as pool:
        for a in pool.imap(load, files, chunksize=16):
            if a is not None: tmp[n] = a; n += 1
    out = np.lib.format.open_memmap(f"{SYN}/clean576.npy", mode="w+", dtype=np.uint8, shape=(n, S, S, 3)); out[:] = tmp[:n]; out.flush(); del tmp; os.remove(f"{SYN}/clean576_tmp.npy")
    c = np.lib.format.open_memmap(f"{SYN}/clean_b.npy", mode="w+", dtype=np.uint8, shape=(n, 512, 512, 3)); c[:] = out[:, 32:544, 32:544]; c.flush()
    print("CLEAN2_DONE", n)
