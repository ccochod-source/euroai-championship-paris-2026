"""Exp 3: how sharp is the "decode the known message on a crop" objective around the true rectangle?
On real train images that are NOT fully watermarked:
  plain rectangles (mask == bounding box): agreement at the GT rect, under shifts / scalings, and after hill climbing
  from a jittered start (IoU of the refined rect with the GT rect).
  object shapes: agreement on the mask bounding box before / after hill climbing (is there a decodable rectangle?).
Usage: python exp3_landscape.py --kit . --data release [--limit N]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--methods", default="trustmark,pixelseal,stegastamp")
a = ap.parse_args()
sys.path.insert(0, a.kit); sys.path.insert(0, str(Path(__file__).parent))
from wmloc.data import LocalisationDataset  # noqa
from wmloc.watermarks import load_watermark  # noqa
from rectlib import Scorer, refine, rect_iou, valid  # noqa

MAXA = {"pixelseal": 16.0, "stegastamp": 16.0, "trustmark": 2.0}
SHIFT = [0.03, 0.06, 0.12, 0.25, 0.5]
SCALE = [0.5, 0.67, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0]
rng = np.random.default_rng(0)


def clip(r):
    y0, y1, x0, x1 = [int(round(v)) for v in r]
    y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(512, y1), min(512, x1)
    return (y0, max(y1, y0 + 8), x0, max(x1, x0 + 8))


for meth in a.methods.split(","):
    wm = load_watermark(meth, models_dir=Path(a.kit) / "models", device="cuda")
    sc = Scorer(wm)
    ds = LocalisationDataset(a.data, "train", methods=[meth])
    idx = [i for i, m in enumerate(ds.meta) if not m.get("full_watermark", False)]
    if a.limit:
        idx = idx[:a.limit]
    R = {"gt": [], "ious": [], "acc_ref": [], "n_eval": [], "obj0": [], "obj1": [], "size": []}
    sh = {f: [] for f in SHIFT}; scl = {f: [] for f in SCALE}
    t0 = time.time()
    for i in idx:
        it = ds[i]; mask = it["mask"]
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        g = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        h, w = g[1] - g[0], g[3] - g[2]
        if min(h, w) < 8:
            continue
        img = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).cuda()
        sm = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda()
        fill = mask[g[0]:g[1], g[2]:g[3]].mean()
        if fill > 0.98:                                             # plain rectangle
            _, acc = sc.score(img, [g], sm); R["gt"].append(acc[0]); R["size"].append(min(h, w))
            for f in SHIFT:
                c = [clip((g[0] + dy * f * h, g[1] + dy * f * h, g[2] + dx * f * w, g[3] + dx * f * w)) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))]
                sh[f].append(sc.score(img, c, sm)[1].mean())
            cy, cx = (g[0] + g[1]) / 2, (g[2] + g[3]) / 2
            for f in SCALE:
                scl[f].append(sc.score(img, [clip((cy - h * f / 2, cy + h * f / 2, cx - w * f / 2, cx + w * f / 2))], sm)[1][0])
            j = clip((g[0] + rng.uniform(-.15, .15) * h, g[1] + rng.uniform(-.15, .15) * h, g[2] + rng.uniform(-.15, .15) * w, g[3] + rng.uniform(-.15, .15) * w))
            n0 = sc.n_eval
            r, s, acc = refine(sc, img, sm, j, max_aspect=MAXA[meth])
            R["ious"].append(rect_iou(r, g)); R["acc_ref"].append(acc); R["n_eval"].append(sc.n_eval - n0)
        else:                                                       # object shape
            _, acc = sc.score(img, [g], sm)
            r, s, acc1 = refine(sc, img, sm, g, max_aspect=MAXA[meth])
            R["obj0"].append(acc[0]); R["obj1"].append(acc1)
    io = np.array(R["ious"]); gt = np.array(R["gt"]); sz = np.array(R["size"])
    print(f"\n===== {meth}: plain rects n={len(gt)}  objects n={len(R['obj0'])}  ({time.time() - t0:.0f}s)")
    print(f"  bit acc at GT rect: mean={gt.mean():.3f}  >0.9: {np.mean(gt > .9):.2f}  >0.75: {np.mean(gt > .75):.2f}   "
          f"by min side <32: {gt[sz < 32].mean() if (sz < 32).any() else float('nan'):.2f} (n={(sz < 32).sum()}), "
          f"32-64: {gt[(sz >= 32) & (sz < 64)].mean() if ((sz >= 32) & (sz < 64)).any() else float('nan'):.2f} (n={((sz >= 32) & (sz < 64)).sum()}), "
          f">=64: {gt[sz >= 64].mean() if (sz >= 64).any() else float('nan'):.2f} (n={(sz >= 64).sum()})")
    print("  acc vs shift (fraction of size): " + "  ".join(f"{f}: {np.mean(v):.2f}" for f, v in sh.items()))
    print("  acc vs scale (about centre):     " + "  ".join(f"{f}: {np.mean(v):.2f}" for f, v in scl.items()))
    print(f"  hill climbing from +-15% jitter: IoU mean={io.mean():.3f} median={np.median(io):.3f}  IoU>0.9: {np.mean(io > .9):.2f}  "
          f"IoU>0.98: {np.mean(io > .98):.2f}  final acc mean={np.mean(R['acc_ref']):.3f}  evals/img={np.mean(R['n_eval']):.0f}")
    good = gt > 0.9
    if good.any():
        print(f"     restricted to rects decodable at GT (acc>0.9, n={good.sum()}): IoU mean={io[good].mean():.3f}  IoU>0.9: {np.mean(io[good] > .9):.2f}")
    if R["obj0"]:
        o0, o1 = np.array(R["obj0"]), np.array(R["obj1"])
        print(f"  objects: acc on mask bbox mean={o0.mean():.3f} -> after hill climbing mean={o1.mean():.3f}  (>0.75: {np.mean(o1 > .75):.2f}, >0.9: {np.mean(o1 > .9):.2f})")
    del wm, sc; torch.cuda.empty_cache()
