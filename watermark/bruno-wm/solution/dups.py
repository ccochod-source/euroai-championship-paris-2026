"""Do the same base images appear several times in the release (different method / mask)? If so we can read the real
watermark residual directly from pairs."""
import json, numpy as np
from PIL import Image
from common import RELEASE, OUT
items = []
for split in ("train", "test"):
    for m in json.load(open(f"{RELEASE}/{split}/metadata.json")):
        items.append((split, m["image_name"], m["method"], m.get("full_watermark")))
th = np.stack([np.asarray(Image.open(f"{RELEASE}/{s}/images/{n}.png").convert("L").resize((24, 24), Image.Resampling.BOX), np.float32).ravel() for s, n, _, _ in items])
d = ((th[:, None, :] - th[None, :, :]) ** 2).mean(-1) ** 0.5
np.fill_diagonal(d, 1e9)
nn = d.min(1); print("nearest-neighbour RMS distance quantiles (0-255 scale):", np.quantile(nn, [0, .01, .05, .25, .5, .75, .95]).round(2))
pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items)) if d[i, j] < 4.0]
print("pairs with distance < 4:", len(pairs)); 
import collections
print(collections.Counter((items[i][0], items[j][0]) for i, j in pairs))
print(collections.Counter(tuple(sorted((items[i][2], items[j][2]))) for i, j in pairs).most_common(8))
for i, j in pairs[:12]: print(items[i], items[j], round(float(d[i, j]), 2))
json.dump([(items[i], items[j], float(d[i, j])) for i, j in pairs], open(f"{OUT}/dup_pairs.json", "w"))
# group sizes
import itertools
parent = list(range(len(items)))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i, j in pairs: parent[find(i)] = find(j)
sizes = collections.Counter(collections.Counter(find(i) for i in range(len(items))).values()); print("group sizes:", sorted(sizes.items()))
