"""Check the batched templater against the kit's own embed (per image)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT  # noqa
from templates import Templater
from wmloc.watermarks import load_watermark
clean = np.load(f"{SYN}/clean.npy", mmap_mode="r")
rng = np.random.default_rng(0)
for meth in ["pixelseal", "trustmark", "stegastamp"]:
    wm = load_watermark(meth, device="cuda"); tp = Templater(wm)
    X = np.stack([np.array(clean[i]) for i in range(4)]); bits = rng.integers(0, 2, (4, wm.nbits))
    ref = np.stack([wm.embed(X[i], bits[i]).astype(np.float32) - X[i] for i in range(4)])
    T = tp(torch.from_numpy(X).permute(0, 3, 1, 2).cuda().float(), torch.from_numpy(bits).cuda()).permute(0, 2, 3, 1).cpu().numpy()
    c = [float(np.corrcoef(ref[i].ravel(), T[i].ravel())[0, 1]) for i in range(4)]
    print(meth, "corr batched-vs-kit:", np.round(c, 4), "| mean abs kit %.3f batched %.3f | mean diff %.3f" % (np.abs(ref).mean(), np.abs(T).mean(), (T - ref).mean()))
    del wm
