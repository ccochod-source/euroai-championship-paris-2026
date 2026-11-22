"""Integrity check of the stored proposals (every train + test image loads, shapes are right)."""
import numpy as np
from common import load_items
from seg.seglib import load_props, partitions
bad = 0; ninst = []
for split in ("train", "test"):
    items = load_items(split)
    for it in items:
        try:
            d = load_props(it["name"], split)
            assert d["slic"].shape == (3, 256, 256) and d["fh"].shape == (3, 256, 256) and d["ws"].shape == (2, 256, 256) and d["ms"].shape == (2, 256, 256)
            assert d["inst"].shape == (len(d["inst_score"]), 256, 256)
            ninst.append(len(d["inst_score"]))
        except Exception as e:
            bad += 1; print("BAD", split, it["name"], repr(e)[:100])
    print(split, len(items), "checked")
ninst = np.array(ninst)
print("bad files:", bad, "| instances per image: mean", ninst.mean().round(1), "min", ninst.min(), "max", ninst.max(), "| images with 0 instance:", int((ninst == 0).sum()))
P = partitions(d); print({k: int(v.max()) + 1 for k, v in P.items()})
