"""Self-calibrated verification. x+ = the image with OUR embedding of the same message added everywhere (kit embedder).
For the true hole H and the same shape moved onto watermarked areas, compare:
  d  = S(x)  - S(x with Q greyed)            (what Q contributes now)
  dp = S(x+) - S(x+ with Q greyed)           (what Q contributes once it is certainly watermarked)
  g  = S(x with Q replaced by x+) - S(x)     (gain when we add our watermark only inside Q: 'fill test')
and report how well d, d/dp, d-dp and g rank the true hole against the moved shapes (1.0 = perfect)."""
import sys, numpy as np, torch
from PIL import Image
from common import *
meth = sys.argv[1]; wm = load_wm(meth); dev = "cuda"; rng = np.random.default_rng(0)
rows = []
for it in [i for i in load_items("train", meth) if i["full"]]:
    x = load_img(it["path"]); sm = sm_of(it["bits"]); H = ~gt512(it["name"]); a = H.mean()
    if a < 0.003: continue
    regs = [H]
    for _ in range(40):
        if len(regs) >= 7: break
        dy, dx = rng.integers(96, 416, 2); Hs = np.roll(np.roll(H, dy, 0), dx, 1)
        if (Hs & H).sum() < 0.1 * H.sum(): regs.append(Hs)
    if len(regs) < 3: continue
    xp = torch.from_numpy(wm.embed(np.asarray(Image.open(it["path"]).convert("RGB"), dtype=np.uint8), it["bits"])).permute(2, 0, 1)[None].float().div(255).to(dev)
    R = torch.from_numpy(np.stack(regs)).to(dev)[:, None].float(); grey = torch.full_like(x, 0.5)
    v = torch.cat([x, xp, x * (1 - R) + grey * R, xp * (1 - R) + grey * R, x * (1 - R) + xp * R]); mg, _ = signed(wm, v, sm)
    n = len(regs); S, Sp = mg[0], mg[1]; d = S - mg[2:2 + n]; dp = Sp - mg[2 + n:2 + 2 * n]; g = mg[2 + 2 * n:] - S
    st = {"d": d, "d/dp": d / np.maximum(dp, 0.02), "d-dp": d - dp, "g": -g, "g/dp": -g / np.maximum(dp, 0.02)}
    rows.append([a, S, Sp, d[0], d[1:].mean(), dp[0], dp[1:].mean(), g[0], g[1:].mean()] + [float((v_[1:] > v_[0]).mean()) for v_ in st.values()])
A = np.array(rows); names = ["d", "d/dp", "d-dp", "g", "g/dp"]
print(f"== {meth} n={len(A)}: margin image {A[:,1].mean():.2f} -> with our watermark added everywhere {A[:,2].mean():.2f}")
print(f"   true hole: d {A[:,3].mean():+.3f} dp {A[:,5].mean():+.3f} g {A[:,7].mean():+.3f} | moved shape: d {A[:,4].mean():+.3f} dp {A[:,6].mean():+.3f} g {A[:,8].mean():+.3f}")
print("   ranking accuracy   " + "  ".join(f"{n}: {A[:, 9 + i].mean():.3f}" for i, n in enumerate(names)))
for lo, hi in [(0.003, .02), (.02, .06), (.06, .12), (.12, 1)]:
    k = (A[:, 0] >= lo) & (A[:, 0] < hi)
    print(f"   hole {lo:.3f}-{hi:.2f} n={k.sum():2d}  " + "  ".join(f"{n}: {A[k, 9 + i].mean():.3f}" for i, n in enumerate(names)) + f" | g true {A[k,7].mean():+.3f} moved {A[k,8].mean():+.3f}")
