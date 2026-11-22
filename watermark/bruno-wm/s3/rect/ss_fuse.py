"""StegaStamp train: when to replace the U-Net mask by the verified rectangle? (CPU only)
Usage: python rect/ss_fuse.py <raw json>[,<raw json>...] [thr]"""
import json, sys
import numpy as np
from common import *
raw = {}
for f in sys.argv[1].split(","):
    raw.update(json.load(open(f"{OUT}/{f}")))
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
SH = int(sys.argv[3]) if len(sys.argv) > 3 else 0          # shrink of the decoded frame, px per side (as finalize.py)
z = np.load(f"{W}/wm/out/stegastamp_ftA_trainprobs.npz"); U = {str(n): p.astype(np.float32) > 0.5 for n, p in zip(z["names"], z["probs"])}
G = gt128(); rows = []
for n, r in raw.items():
    if r.get("full") is not False:
        continue
    gt = G[n]; u = U[n]; row = dict(name=n, kind=r["kind"], unet=miou(u, gt), acc=r.get("acc", 0), ok=r.get("acc", 0) >= thr)
    if row["ok"]:
        m = np.zeros((512, 512), np.float32); y0, y1, x0, x1 = r["rect"]
        if SH and min(y1 - y0, x1 - x0) > 4 * SH:
            y0, y1, x0, x1 = y0 + SH * (y0 > 0), y1 - SH * (y1 < 512), x0 + SH * (x0 > 0), x1 - SH * (x1 < 512)
        m[y0:y1, x0:x1] = 1; rm = downsample_mask(m) > 0
        row.update(rect=miou(rm, gt), fill=float((u & rm).sum() / max(1, rm.sum())), spill=float((u & ~rm).sum() / max(1, u.sum())),
                   inter=miou(u & rm, gt), union=miou(u | rm, gt), rm=rm, u=u)
        if len(sys.argv) > 4: print(f"{n} {r['kind']:5s} acc {row['acc']:.2f} unet {row['unet']:.3f} rect {row['rect']:.3f} inter {row['inter']:.3f} union {row['union']:.3f} fill {row['fill']:.2f} spill {row['spill']:.2f}")
    rows.append(row)


def total(rule):
    out = {}
    for k in ("rect", "shape"):
        out[k] = np.mean([rule(r) if r["ok"] else r["unet"] for r in rows if r["kind"] == k])
    return "rect %.3f shape %.3f  all not-full %.3f" % (out["rect"], out["shape"], np.mean([rule(r) if r["ok"] else r["unet"] for r in rows]))


print("U-Net only            ", total(lambda r: r["unet"]))
print("rect when accepted    ", total(lambda r: r["rect"]))
print("oracle max            ", total(lambda r: max(r["unet"], r["rect"])))
print("intersection          ", total(lambda r: r["inter"]))
for f in (0.5, 0.7, 0.8, 0.9):
    print(f"rect if fill >= {f} else unet&rect ", total(lambda r: r["rect"] if r["fill"] >= f else r["inter"]))
    print(f"rect if fill >= {f} else unet      ", total(lambda r: r["rect"] if r["fill"] >= f else r["unet"]))
for s in (0.02, 0.1, 0.3):
    print(f"unet if fill>=0.9 and spill<={s} else rect", total(lambda r: r["unet"] if (r["fill"] >= 0.9 and r["spill"] <= s) else r["rect"]))
