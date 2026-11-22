"""How much localisation does the value-histogram cue give on the challenge train images?
LLR(v) = log P(v | clean) / P(v | watermarked), estimated per channel on half of the train images and tested on the
other half. Evidence map = box-filtered sum of LLR; reported: AUC of cells (clean vs watermarked) per image, and the
mIoU of a thresholded map (threshold tuned on the fitting half)."""
import json, numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from common import RELEASE
from wmloc.utils import miou_image, downsample_mask
meta = json.load(open(f"{RELEASE}/train/metadata.json"))
def load(m):
    a = np.asarray(Image.open(f"{RELEASE}/train/images/{m['image_name']}.png").convert("RGB"))
    M = np.asarray(Image.open(f"{RELEASE}/train/masks/{m['image_name']}.png")) > 127
    return a, M
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
for meth in ("trustmark", "pixelseal", "stegastamp"):
    items = [m for m in meta if m["method"] == meth]; data = [load(m) for m in items]
    res = {}
    for half in (0, 1):
        fit = [d for i, d in enumerate(data) if i % 2 == half]; tst = [(items[i], d) for i, d in enumerate(data) if i % 2 != half]
        Hc = np.ones((3, 256)); Hw = np.ones((3, 256))
        for a, M in fit:
            for ch in range(3):
                Hc[ch] += np.bincount(a[..., ch][~M], minlength=256); Hw[ch] += np.bincount(a[..., ch][M], minlength=256)
        # keep only the fine structure (comb): ratio to a 5-wide local mean, so that image content (brightness) does not leak
        def fine(H): return H / np.stack([np.convolve(h, np.ones(5) / 5, mode="same") for h in H])
        L = np.log(fine(Hc)) - np.log(fine(Hw)); L[:, :3] = 0; L[:, -3:] = 0
        for scale in (8, 16, 32, 64):
            maps = []
            for it, (a, M) in tst:
                e = sum(L[ch][a[..., ch]] for ch in range(3)); e = uniform_filter(e, size=scale, mode="reflect") * scale * scale
                maps.append(e.reshape(128, 4, 128, 4).mean((1, 3)))
            # evidence > 0 means clean. per-image AUC
            aucs = []; 
            for (it, (a, M)), e in zip(tst, maps):
                y = downsample_mask(M.astype(np.float32)).astype(bool).ravel()
                if 50 < y.sum() < y.size - 50: aucs.append(auc(-e.ravel(), y))
            res.setdefault(scale, []).append(np.mean(aucs))
    print(f"{meth}: mean per-image AUC (watermarked vs clean cells) by smoothing scale: " + "  ".join(f"{s}px {np.mean(v):.3f}" for s, v in res.items()), flush=True)
