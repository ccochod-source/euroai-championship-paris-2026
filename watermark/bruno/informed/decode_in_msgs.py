"""Recover the messages of the other session's in-domain fully watermarked images ({method}_in_full.npy, N x 3 x 512 x 512 x 3)
by decoding them (they were embedded with random messages that were not stored)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT  # noqa
from wmloc.watermarks import load_watermark
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(out, exist_ok=True)
for meth in sys.argv[1:] or ["pixelseal", "trustmark", "stegastamp"]:
    wm = load_watermark(meth, device="cuda"); F = np.load(f"{SYN}/{meth}_in_full.npy", mmap_mode="r"); N, K = F.shape[:2]
    msgs = np.zeros((N, K, wm.nbits), np.uint8); conf = np.zeros((N, K), np.float32)
    for i in range(N):
        x = torch.from_numpy(np.array(F[i])).permute(0, 3, 1, 2).float().div(255)
        lg = wm.logits(x).float().cpu().numpy(); msgs[i] = lg > 0; conf[i] = np.abs(lg).min(1)
    np.save(f"{out}/{meth}_in_msgs.npy", msgs)
    # check on one image: re-embed the base with the decoded message and compare with the stored watermarked image
    B = np.load(f"{SYN}/{meth}_in_base.npy", mmap_mode="r"); d = [float(np.abs(wm.embed(np.array(B[i]), msgs[i, 0]).astype(int) - np.array(F[i, 0])).mean()) for i in range(3)]
    print(meth, "N", N, "K", K, "min |logit| quantiles", np.quantile(conf, [0, .01, .1, .5]).round(2), "| re-embed mean abs diff (should be ~0):", np.round(d, 4), flush=True)
    del wm; torch.cuda.empty_cache()
