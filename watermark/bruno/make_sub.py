"""Build submission.pt from per-position maps: per-method threshold fit on train (max mIoU), applied on test.
Usage: python make_sub.py --train maps_train2.npz --test maps_test2.npz --map pp --out output/sub_pp.pt [--submit]
API key read from ~/.hack_api_key.
"""
import argparse
from pathlib import Path

import numpy as np
import requests
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--map", default="pp"); ap.add_argument("--out", required=True)
ap.add_argument("--submit", action="store_true")
a = ap.parse_args()


def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = np.logical_or(p == c, g == c).sum()
        s += 1.0 if u == 0 else np.logical_and(p == c, g == c).sum() / u
    return s / 2


tr, te = np.load(a.train), np.load(a.test)
key = "map_" + a.map
thr, tot = {}, []
for m in np.unique(tr["method"]):
    sel = tr["method"] == m
    M, G = tr[key][sel], tr["gt"][sel]
    qs = np.nanquantile(M, np.linspace(0.02, 0.98, 97))
    sc, t = max((np.mean([miou((x > t).astype(np.uint8), y) for x, y in zip(M, G)]), t) for t in qs)
    thr[m] = t; tot += [sc] * sel.sum()
    print(f"{m:11s} T={t:.4f} train mIoU={sc:.4f}")
print(f"train overall {np.mean(tot):.4f}")

names = [str(n) for n in te["names"]]
masks = np.stack([(x > thr[m]).astype(np.uint8) for x, m in zip(te[key], te["method"])])
assert masks.shape == (len(names), 128, 128) and len(set(names)) == len(names), masks.shape
assert not np.isnan(te[key]).any()
torch.save({"image_names": names, "masks": torch.from_numpy(masks).bool()}, a.out)
print("wrote", a.out, masks.shape, f"{Path(a.out).stat().st_size / 1e6:.1f} MB", "pos frac", masks.mean().round(3))

if a.submit:
    api = (Path.home() / ".hack_api_key").read_text().strip()
    with open(a.out, "rb") as f:
        r = requests.post("http://34.147.229.55/submit/34-watermark-localisation-image", headers={"X-API-Key": api},
                          files={"file": (Path(a.out).name, f, "application/octet-stream")}, timeout=(10, 600))
    print(r.status_code, r.text[:500])
