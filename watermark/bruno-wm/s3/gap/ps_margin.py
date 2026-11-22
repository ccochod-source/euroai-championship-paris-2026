"""Decoder-based look at the frame of real full images: margin of the known message when the image is placed in a
larger square canvas (side 512 + e) at a few offsets, the rest filled by reflection / grey / edge replication.
If the true frame is larger than the visible image, the margin should rise at the right (e, offset).
usage: ps_margin.py <method> [n]"""
import sys, numpy as np, torch, torch.nn.functional as F
from common import *
meth = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
wm = load_wm(meth); dev = "cuda"
ES = [0, 8, 16, 32, 48, 64, 96, 128, 192, 256]

def place(x, e, t, l, fill):
    if e == 0: return x
    if fill == "grey":
        c = torch.full((1, 3, 512 + e, 512 + e), 0.5, device=x.device); c[:, :, t:t + 512, l:l + 512] = x; return c
    return F.pad(x, (l, e - l, t, e - t), mode=("reflect" if fill == "reflect" and e < 511 else "replicate"))

items = [it for it in load_items("train", meth) if it["full"]][:n]
for fill in ["reflect", "grey", "replicate"]:
    R = []
    for it in items:
        x = load_img(it["path"]); sm = sm_of(it["bits"]); row = []
        for e in ES:
            offs = [(e // 2, e // 2)] if e == 0 else [(t, l) for t in (0, e // 2, e) for l in (0, e // 2, e)]
            imgs = torch.cat([F.interpolate(place(x, e, t, l, fill), size=(512, 512), mode="bilinear", align_corners=False, antialias=True) for t, l in offs])
            m, _ = signed(wm, imgs, sm); row.append((m.max(), m.mean()))
        R.append(row)
    R = np.array(R)
    print(f"== {meth} fill={fill} n={len(items)}: margin by canvas extension e (best of 9 offsets | mean over offsets)")
    print("   e        " + " ".join(f"{e:6d}" for e in ES))
    print("   best     " + " ".join(f"{v:6.2f}" for v in R[:, :, 0].mean(0)))
    print("   mean     " + " ".join(f"{v:6.2f}" for v in R[:, :, 1].mean(0)))
    if fill == "reflect":
        arg = R[:, :, 0].argmax(1); print("   per image, e with the highest margin: " + " ".join(f"{ES[k]}({R[i, k, 0] / max(R[i, 0, 0], 1e-3):.2f}x)" for i, k in enumerate(arg)))
