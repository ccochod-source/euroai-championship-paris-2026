"""Visual check: image + ground-truth contour (red) | fh1 boundaries | Mask R-CNN masks, for a few train images."""
import json
import sys

import cv2
import numpy as np

from common import OUT, gt512, load_items
from seg.seglib import load_props, up2

res = {r["name"]: r for r in json.load(open(f"{OUT}/seg/oracle.json"))}
items = load_items("train")
rows = []
for grp in ("full", "shape"):
    sel = [it for it in items if res[it["name"]]["group"] == grp][3:60:8][:7]
    for it in sel:
        img = cv2.imread(it["path"]); gt = gt512(it["name"]).astype(np.uint8)
        d = load_props(it["name"])
        cnt, _ = cv2.findContours(gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        a = img.copy(); cv2.drawContours(a, cnt, -1, (0, 0, 255), 2)
        b = img.copy(); lab = up2(d["fh"][1].astype(np.int32))
        edge = (lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1)); b[edge] = (0, 255, 255); cv2.drawContours(b, cnt, -1, (0, 0, 255), 2)
        c = (img * 0.4).astype(np.uint8)
        rng = np.random.default_rng(0)
        for m, s in list(zip(d["inst"], d["inst_score"]))[:12][::-1]:
            col = rng.integers(60, 255, 3); mm = up2(m); c[mm] = (0.5 * c[mm] + 0.5 * col).astype(np.uint8)
        cv2.drawContours(c, cnt, -1, (0, 0, 255), 2)
        r = res[it["name"]]["s"]
        cv2.putText(a, f"{grp} {it['method'][:5]} inst {r['inst'][0]:.2f}/{r['inst'][1]:.2f} fh1 {r['fh1'][0]:.2f} g16 {r['grid16'][0]:.2f}", (4, 18), 0, 0.5, (255, 255, 255), 2)
        rows.append(np.concatenate([a, b, c], 1))
cv2.imwrite(f"{OUT}/seg/viz_full.jpg", cv2.resize(np.concatenate(rows[:7], 0), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
cv2.imwrite(f"{OUT}/seg/viz_shape.jpg", cv2.resize(np.concatenate(rows[7:], 0), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("ok", len(rows))
