"""JPEG history test: in a JPEG-decoded image the 8x8 block DCT coefficients cluster on multiples of the quantisation
step q. Comb score = mean cos(2*pi*coef/q) over blocks (0 for never-compressed images). Computed separately for blocks
fully inside the watermarked region and fully inside the clean region of the challenge train images."""
import io, json, numpy as np
from PIL import Image
from scipy.fft import dctn
from common import RELEASE
meta = json.load(open(f"{RELEASE}/train/metadata.json"))
def blocks(a):
    y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float64) - 128
    b = y.reshape(64, 8, 64, 8).transpose(0, 2, 1, 3)
    return dctn(b, axes=(2, 3), norm="ortho")                      # [64, 64, 8, 8]
def comb(c, q):
    return float(np.cos(2 * np.pi * c / q).mean())
QS = [2, 3, 4, 5, 6, 8, 10, 12, 16]
rows = {}
for m in meta[:240]:
    a = np.asarray(Image.open(f"{RELEASE}/train/images/{m['image_name']}.png").convert("RGB"))
    M = (np.asarray(Image.open(f"{RELEASE}/train/masks/{m['image_name']}.png")) > 127).reshape(64, 8, 64, 8).mean((1, 3))
    D = blocks(a)
    for reg, sel in (("wm", M == 1), ("clean", M == 0)):
        if sel.sum() < 200: continue
        for cf in ((0, 1), (1, 1), (2, 2), (0, 4)):
            c = D[sel][:, cf[0], cf[1]]
            c = c[np.abs(c) > 1.0]                                  # ignore the pile at zero
            if len(c) < 100: continue
            rows.setdefault((m["method"], reg, cf), []).append([comb(c, q) for q in QS])
    if len(rows) and m is meta[0]: pass
print("comb score per quantisation step q =", QS)
for k in sorted(rows): print(k, "n=%d" % len(rows[k]), " ".join("%+.2f" % v for v in np.mean(rows[k], 0)))
# reference: what a JPEG round trip looks like on a challenge image
a = np.asarray(Image.open(f"{RELEASE}/train/images/{meta[0]['image_name']}.png").convert("RGB"))
for Q in (75, 90, 95, 98):
    buf = io.BytesIO(); Image.fromarray(a).save(buf, format="JPEG", quality=Q); j = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    c = blocks(j)[:, :, 1, 1].ravel(); c = c[np.abs(c) > 1.0]
    print(f"reference JPEG Q={Q} coef(1,1):", " ".join("%+.2f" % comb(c, q) for q in QS))
