"""Sanity check of the 'tall canvas' model with the decoder. (1) Simulation: take real images whose plain frame matches,
re-watermark them with the kit on a tall canvas (reflect padding), crop back 512 rows, and decode the crop: does the
decoder still read it? (2) Real images with a large estimated canvas: decoder margin on the visible image versus on
the reflect-padded canvas."""
import sys, json, numpy as np, torch
from PIL import Image
from common import *
meth = sys.argv[1]; wm = load_wm(meth); dev = "cuda"; rng = np.random.default_rng(0)
cv = json.load(open(f"{OUT}/canvas_{meth}_train.json")); items = {it["name"]: it for it in load_items("train", meth)}
def pad(u8, a, b): return np.ascontiguousarray(np.pad(u8, ((a, b), (0, 0), (0, 0)), mode="reflect"))
def t(u8): return torch.from_numpy(u8).permute(2, 0, 1)[None].float().div(255).to(dev)
print(f"== {meth} (1) simulation: kit watermark on a tall canvas, decode the 512-row crop. columns: hidden rows (above, below) -> margin / bit accuracy of the crop")
base = [k for k, v in cv.items() if v["a"] + v["b"] <= 4][:6]
CFG = [(0, 0), (16, 16), (32, 0), (40, 40), (100, 28), (128, 128), (272, 0), (200, 200)]
R = []
for n in base:
    u8 = np.asarray(Image.open(items[n]["path"]).convert("RGB"), dtype=np.uint8); row = []
    for a, b in CFG:
        bits = rng.integers(0, 2, len(items[n]["bits"])); w = wm.embed(pad(u8, a, b), bits)[a:a + 512]
        m, ac = signed(wm, t(np.ascontiguousarray(w)), sm_of(bits)); row.append((m[0], ac[0]))
    R.append(row)
R = np.array(R)
print("   " + "  ".join(f"({a},{b}): {R[:, i, 0].mean():+.2f}/{R[:, i, 1].mean():.2f}" for i, (a, b) in enumerate(CFG)))
print(f"== {meth} (2) real images: margin / accuracy on the visible image vs on the estimated canvas (reflect-padded)")
big = sorted(cv, key=lambda k: -(cv[k]["a"] + cv[k]["b"]))[:8] + [k for k, v in cv.items() if 20 <= v["a"] + v["b"] <= 80][:8]
for n in big:
    it = items[n]; u8 = np.asarray(Image.open(it["path"]).convert("RGB"), dtype=np.uint8); c = cv[n]
    m0, a0 = signed(wm, t(u8), sm_of(it["bits"])); m1, a1 = signed(wm, t(pad(u8, c["a"], c["b"])), sm_of(it["bits"]))
    print(f"   {n} canvas ({c['a']},{c['b']}) slope {c['slope']:+.2f} plain {c['slope_plain']:+.2f} | visible {m0[0]:+.2f}/{a0[0]:.2f} | canvas {m1[0]:+.2f}/{a1[0]:.2f}")
