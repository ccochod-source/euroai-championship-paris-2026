"""Which frame carries the watermark in not-full SHAPE images? And does occlusion behave additively on full images?
For each real training image we decode several variants and report the decoder margin / bit accuracy:
  whole      : the image as is (frame = whole image)
  keep       : everything outside the ground-truth mask greyed (only watermarked pixels left), frame = whole image
  kill       : the ground-truth mask greyed (watermark removed), frame = whole image
  bbox       : crop to the mask bounding box
  bbox_keep  : crop to the bounding box, pixels outside the mask greyed
"""
import sys, numpy as np, torch
from common import *

def variants(img, m512):
    m = torch.from_numpy(m512).to(img.device)[None, None].float()
    grey = torch.full_like(img, 0.5)
    ys, xs = np.nonzero(m512); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    keep = img * m + grey * (1 - m); kill = img * (1 - m) + grey * m
    rs = lambda t: torch.nn.functional.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False, antialias=True)
    return torch.cat([img, keep, kill, rs(img[:, :, y0:y1, x0:x1]), rs(keep[:, :, y0:y1, x0:x1])])

NAMES = ["whole", "keep", "kill", "bbox", "bbox_keep"]
for meth in sys.argv[1:]:
    wm = load_wm(meth); g = gt128(); rows = {}
    for it in load_items("train", meth):
        m512 = gt512(it["name"])
        if m512.sum() < 64: continue
        k = "full" if it["full"] else kind(g[it["name"]])
        mg, ac = signed(wm, variants(load_img(it["path"]), m512), sm_of(it["bits"]))
        rows.setdefault(k, []).append(np.concatenate([mg, ac, [m512.mean()]]))
    for k, R in rows.items():
        R = np.array(R); print(f"== {meth} {k:5s} n={len(R)}  median wm area {np.median(R[:, -1]):.3f}")
        for j, nm in enumerate(NAMES):
            print(f"   {nm:10s} margin mean {R[:, j].mean():+6.2f} median {np.median(R[:, j]):+6.2f} | acc mean {R[:, 5 + j].mean():.3f}  share acc>0.75: {(R[:, 5 + j] > 0.75).mean():.2f}  >0.65: {(R[:, 5 + j] > 0.65).mean():.2f}")
        if k == "shape":
            a = R[:, -1]; o = np.argsort(a)
            for lo, hi in [(0, len(o) // 3), (len(o) // 3, 2 * len(o) // 3), (2 * len(o) // 3, len(o))]:
                S = R[o[lo:hi]]; print(f"   area tercile {S[:, -1].min():.3f}-{S[:, -1].max():.3f}: acc whole {S[:, 5].mean():.3f} keep {S[:, 6].mean():.3f} kill {S[:, 7].mean():.3f} bbox {S[:, 8].mean():.3f} bbox_keep {S[:, 9].mean():.3f}")
