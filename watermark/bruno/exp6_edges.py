"""Exp 6: shape of the decoding objective when ONE edge of the true rectangle is moved (real train, plain rectangles).
Prints, per method, the mean objective vs edge offset (relative to the rect side), for two objectives:
raw margin = mean(sign * logit) and bit accuracy. Negative offset = rectangle grows beyond the true edge.
Usage: python exp6_edges.py --kit . --data release --methods trustmark,pixelseal,stegastamp
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--methods", default="trustmark,pixelseal,stegastamp")
a = ap.parse_args()
sys.path.insert(0, a.kit); sys.path.insert(0, str(Path(__file__).parent))
from wmloc.data import LocalisationDataset  # noqa
from wmloc.watermarks import load_watermark  # noqa
from rectlib import Scorer  # noqa

OFF = [-0.3, -0.2, -0.12, -0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3]
for meth in a.methods.split(","):
    wm = load_watermark(meth, models_dir=Path(a.kit) / "models", device="cuda"); sc = Scorer(wm)
    ds = LocalisationDataset(a.data, "train", methods=[meth])
    M, A = [], []
    for i, md in enumerate(ds.meta):
        if md.get("full_watermark", False):
            continue
        it = ds[i]; mask = it["mask"]; ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        g = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]
        h, w = g[1] - g[0], g[3] - g[2]
        if mask[g[0]:g[1], g[2]:g[3]].mean() < 0.98 or min(h, w) < 40:
            continue
        img = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).cuda()
        sm = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda()
        rects, keep = [], []
        for e in range(4):                                   # 0: top, 1: bottom, 2: left, 3: right
            side = h if e < 2 else w
            for o in OFF:
                r = list(g); d = int(round(o * side))
                r[e] = r[e] + d if e in (0, 2) else r[e] - d  # positive offset shrinks the rectangle
                ok = 0 <= r[0] < r[1] <= 512 and 0 <= r[2] < r[3] <= 512 and r[1] - r[0] >= 8 and r[3] - r[2] >= 8
                rects.append(tuple(r) if ok else tuple(g)); keep.append(ok)
        lg = sc.logits_res(sc.crops(img, rects))
        marg = (lg * sm[None]).mean(1).cpu().numpy(); acc = ((lg > 0).float() * 2 - 1 == sm[None]).float().mean(1).cpu().numpy()
        keep = np.array(keep).reshape(4, len(OFF)); marg = marg.reshape(4, len(OFF)); acc = acc.reshape(4, len(OFF))
        if acc[0, OFF.index(0.0)] < 0.9:
            continue
        marg = np.where(keep, marg, np.nan) / marg[0, OFF.index(0.0)]; acc = np.where(keep, acc, np.nan)
        M.append(marg); A.append(acc)
    M, A = np.array(M), np.array(A)
    print(f"\n===== {meth}: {len(M)} decodable plain rectangles (min side >= 40)")
    print("  offset/side :   " + "  ".join(f"{o:+.2f}" for o in OFF))
    print("  rel. margin :   " + "  ".join(f"{v:5.2f}" for v in np.nanmean(M, (0, 1))))
    print("  bit accuracy:   " + "  ".join(f"{v:5.2f}" for v in np.nanmean(A, (0, 1))))
    best = np.nanargmax(np.where(np.isnan(M), -9, M), 2)
    print("  argmax of the margin over offsets (share of edges): " + "  ".join(f"{OFF[k]:+.2f}:{np.mean(best == k):.2f}" for k in range(len(OFF))))
    del wm, sc; torch.cuda.empty_cache()
