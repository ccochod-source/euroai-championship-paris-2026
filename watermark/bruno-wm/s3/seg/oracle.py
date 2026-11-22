"""Oracle test on the 510 real training images: best mask obtainable from whole segments, official metric.
Usage: ./sg 1 seg/oracle.py [ncpu]   -> prints tables, saves <OUT>/seg/oracle.json"""
import json
import sys

import numpy as np

from common import OUT, downsample_mask, gt128, gt512, kind, load_items, miou
from seg.seglib import grid, idle_pool, load_props, oracle_inst_union, oracle_partition, partitions, refine

COMBOS = [("inst", "fh1"), ("inst", "fh2"), ("inst", "slic1"), ("inst", "slic2"), ("inst", "ms1"), ("inst", "ws1"),
          ("fh2", "slic2"), ("fh2", "ms1"), ("inst", "fh2", "slic2"), ("inst", "fh2", "ms1", "ws1", "slic2")]
G = gt128()


def one(it):
    name = it["name"]
    gt = gt512(name); g = G[name]
    d = load_props(name)
    P = partitions(d)
    for n in (4, 8, 16, 32, 64):
        P[f"grid{n}"] = grid(n)
    for c in COMBOS:
        lab = P[c[0]]
        for k in c[1:]:
            lab = refine(lab, P[k])
        P["+".join(c)] = lab
    r = dict(name=name, method=it["method"], group="full" if it["full"] else kind(g), area=float(g.mean()),
             gt_mismatch=int((downsample_mask(gt) != g).sum()), n_inst=int(len(d["inst"])), s={}, n={})
    r["s"]["all0"] = [miou(np.zeros_like(g), g)] * 2; r["s"]["all1"] = [miou(np.ones_like(g), g)] * 2
    r["n"]["all0"] = r["n"]["all1"] = 1
    for k, lab in P.items():
        r["s"][k] = [miou(downsample_mask(oracle_partition(lab, gt, box)), g) for box in (False, True)]
        r["n"][k] = int(lab.max()) + 1
    u = oracle_inst_union(d["inst"], gt)
    r["s"]["inst_union(no bg)"] = [miou(downsample_mask(u), g)] * 2; r["n"]["inst_union(no bg)"] = len(d["inst"])
    return r


def main():
    pool = idle_pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    res = pool.map(one, load_items("train"), chunksize=4)
    json.dump(res, open(f"{OUT}/seg/oracle.json", "w"))
    groups = ["full", "shape", "rect"]
    print("images:", {g: sum(r["group"] == g for r in res) for g in sorted({r["group"] for r in res})},
          "| cells where downsample_mask(gt512) != masks_128:", sum(r["gt_mismatch"] for r in res))
    keys = list(res[0]["s"])

    def mean(k, box, grp=None, meth=None):
        v = [r["s"][k][box] for r in res if (grp is None or r["group"] == grp) and (meth is None or r["method"] == meth)]
        return np.mean(v) if v else float("nan")

    print("\nORACLE mIoU (official metric). 'box' = segments clipped to the ground-truth bounding rectangle.")
    print(f"{'source':34s} {'#seg':>6s} | {'full':>6s} {'shape':>6s} {'rect':>6s} | {'shape+box':>9s} {'rect+box':>8s} | {'all':>6s}")
    for k in keys:
        n = np.mean([r["n"][k] for r in res])
        print(f"{k:34s} {n:6.0f} | " + " ".join(f"{mean(k, 0, g):6.3f}" for g in groups)
              + f" | {mean(k, 1, 'shape'):9.3f} {mean(k, 1, 'rect'):8.3f} | {mean(k, 0):6.3f}")
    print("\nPER METHOD (no box): columns = method x group")
    ms = ["pixelseal", "trustmark", "stegastamp"]
    print(f"{'source':34s} | " + " | ".join(f"{m[:5]:>5s}:full shape  rect" for m in ms))
    for k in keys:
        print(f"{k:34s} | " + " | ".join(" ".join(f"{mean(k, 0, g, m):10.3f}" if g == "full" else f"{mean(k, 0, g, m):5.3f}"
                                                   for g in groups) for m in ms))
    print("\ncounts per method x group:", {m: {g: sum(r["method"] == m and r["group"] == g for r in res) for g in groups} for m in ms})
    # hole statistics on full images: how many holes does the best partition recover?
    for k in ("inst", "fh1", "inst+fh2"):
        v = np.array([r["s"][k][0] for r in res if r["group"] == "full"])
        print(f"full images, {k}: mean {v.mean():.3f} median {np.median(v):.3f} share > 0.9: {(v > 0.9).mean():.2f} share < 0.6: {(v < 0.6).mean():.2f}")
    for k in ("inst", "fh1", "inst+fh2"):
        v = np.array([r["s"][k][1] for r in res if r["group"] == "shape"])
        print(f"shape images (+box), {k}: mean {v.mean():.3f} median {np.median(v):.3f} share > 0.9: {(v > 0.9).mean():.2f} share < 0.6: {(v < 0.6).mean():.2f}")


if __name__ == "__main__":
    main()
