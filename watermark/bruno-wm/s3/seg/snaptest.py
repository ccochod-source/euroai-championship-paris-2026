"""Test of seglib.snap on the 510 real training images with a synthetic evidence map: ground truth averaged on
16 x 16 cells (32 px each), + Gaussian noise per cell, bilinearly upsampled to 128 x 128.
Usage: ./sg 1 seg/snaptest.py [ncpu]"""
import json
import sys
import zlib

import cv2
import numpy as np

from common import OUT, gt128, kind, load_items, miou
from seg.seglib import grid, idle_pool, load_props, partitions, refine, snap, snap_fit

G = gt128()
NOISE = (0.0, 0.15, 0.3)
SINGLE = ["fh2", "slic1", "slic2", "ms1", "ws1", "inst", "inst+fh1", "inst+fh2", "grid16"]
MULTI = {"fh0>fh1>fh2": ["fh0", "fh1", "fh2"], "slic0>slic1>slic2": ["slic0", "slic1", "slic2"],
         "inst>inst+fh1>inst+fh2": ["inst", "inst+fh1", "inst+fh2"], "inst>fh1>fh2": ["inst", "fh1", "fh2"],
         "ms0>ms1>fh2": ["ms0", "ms1", "fh2"]}
MARGINS = (0.35,)
FIT = ["fh2", "slic1", "slic2", "ms1", "ws1", "inst+fh1", "inst+fh2", "inst+slic1", "grid16", "grid32"]


def degrade(g, sigma, rng):
    c = g.reshape(16, 8, 16, 8).mean((1, 3)).astype(np.float32)
    c = c + rng.normal(0, sigma, c.shape).astype(np.float32)
    return cv2.resize(c, (128, 128), interpolation=cv2.INTER_LINEAR)


def one(it):
    name = it["name"]; g = G[name]
    P = partitions(load_props(name))
    P["inst+fh1"] = refine(P["inst"], P["fh1"]); P["inst+fh2"] = refine(P["inst"], P["fh2"])
    P["inst+slic1"] = refine(P["inst"], P["slic1"]); P["grid16"] = grid(16); P["grid32"] = grid(32)
    r = dict(group="full" if it["full"] else kind(g), method=it["method"], s={})
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    for sg in NOISE:
        ev = degrade(g, sg, rng)
        s = {"evidence>0.5 (no snap)": miou(ev > 0.5, g)}
        for k in SINGLE:
            s[f"hard {k}"] = miou(snap(ev, P[k]), g)
            s[f"soft.25 {k}"] = miou(snap(ev, P[k], margin=0.25), g)
        for k in FIT:
            s[f"fit {k}"] = miou(snap_fit(ev, P[k]), g)
        for k, ks in MULTI.items():
            for m in MARGINS:
                s[f"multi{m:.2f} {k}"] = miou(snap(ev, [P[x] for x in ks], margin=m), g)
        r["s"][str(sg)] = s
    return r


def main():
    pool = idle_pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    res = pool.map(one, load_items("train"), chunksize=4)
    json.dump(res, open(f"{OUT}/seg/snaptest.json", "w"))
    groups = ["full", "shape", "rect"]
    for sg in NOISE:
        print(f"\nnoise sigma {sg} on the 16x16 cells | mIoU: full shape rect | all")
        keys = list(res[0]["s"][str(sg)])
        rows = []
        for k in keys:
            v = [np.mean([r["s"][str(sg)][k] for r in res if r["group"] == g]) for g in groups]
            rows.append((k, v, np.mean([r["s"][str(sg)][k] for r in res])))
        for k, v, a in rows:
            print(f"{k:34s} " + " ".join(f"{x:.3f}" for x in v) + f" | {a:.3f}")


if __name__ == "__main__":
    main()
