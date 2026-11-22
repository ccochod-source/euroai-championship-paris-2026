"""Exp 9: do several dataset images share the same base image (same scene with different watermark / edit)?
Thumbnail nearest neighbours over train + test, then exact pixel comparison of the closest pairs."""
import json, sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
R = Path(sys.argv[1]); items = []
for split in ("train", "test"):
    for m in json.load(open(R / split / "metadata.json")):
        items.append((split, m["image_name"], m["method"], m.get("full_watermark")))
def load(split, name): return np.asarray(Image.open(R / split / "images" / f"{name}.png").convert("RGB"))
th = []
for s, n, _, _ in items:
    g = cv2.resize(cv2.cvtColor(load(s, n), cv2.COLOR_RGB2GRAY), (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g = g - g.mean(); th.append(g / (np.linalg.norm(g) + 1e-6))
th = np.stack(th); S = th @ th.T; np.fill_diagonal(S, -1)
nn = S.argmax(1); best = S.max(1)
print("images:", len(items), "| nearest-neighbour thumbnail correlation quantiles:", np.quantile(best, [.01, .1, .5, .9, .99]).round(3))
for thr in (0.999, 0.99, 0.97, 0.95, 0.9):
    print(f"  images with a neighbour at corr > {thr}: {(best > thr).sum()}")
order = np.argsort(-best)[:12]
for i in order:
    j = nn[i]; A, B = load(*items[i][:2]).astype(np.int16), load(*items[j][:2]).astype(np.int16); d = np.abs(A - B).max(2)
    print(f"  {items[i]} ~ {items[j]} corr={best[i]:.4f}  identical px={np.mean(d == 0):.3f}  |diff|<=3: {np.mean(d <= 3):.3f}  mean abs diff={d.mean():.2f}")
np.save(sys.argv[2], S.astype(np.float16))
