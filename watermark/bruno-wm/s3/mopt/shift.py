"""Feasibility of segment-level verification. For each real full image: overwrite the TRUE hole (ground truth) with a
fill and measure the margin drop; do the same with the same shape moved elsewhere (a watermarked region).
If the two drops separate, 'propose a region, verify with the decoder' works whenever the region is right.
usage: shift.py <method> ; fills: grey, blur (sigma 12), removed (trustmark remover)"""
import sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import *
from scipy.stats import rankdata
meth = sys.argv[1]; wm = load_wm(meth); dev = "cuda"; rng = np.random.default_rng(0)
def blur(x, s):
    k = int(4 * s) | 1; ax = torch.arange(k, device=x.device) - k // 2; g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
def auc(pos, neg):
    s = np.concatenate([pos, neg]); r = rankdata(s); n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
fills = ["grey", "blur"] + (["removed"] if meth == "trustmark" else [])
res = {f: [] for f in fills}
for it in [i for i in load_items("train", meth) if i["full"]]:
    x = load_img(it["path"]); sm = sm_of(it["bits"]); H = ~gt512(it["name"]); a = H.mean()
    if a < 0.003: continue
    regs = [H]
    for _ in range(40):
        if len(regs) >= 7: break
        dy, dx = rng.integers(96, 416, 2); Hs = np.roll(np.roll(H, dy, 0), dx, 1)
        if (Hs & H).sum() < 0.1 * H.sum(): regs.append(Hs)
    if len(regs) < 3: continue
    R = torch.from_numpy(np.stack(regs)).to(dev)[:, None].float()
    for f in fills:
        if f == "grey": fl = torch.full_like(x, 0.5)
        elif f == "blur": fl = blur(x, 12.0)
        else:
            o = wm.tm.remove_watermark(Image.open(it["path"]).convert("RGB")); fl = torch.from_numpy(np.asarray(o.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1)[None].float().div(255).to(dev)
        v = torch.cat([x, x * (1 - R) + fl * R]); mg, _ = signed(wm, v, sm)
        d = mg[0] - mg[1:]
        res[f].append((a, mg[0], d[0], d[1:].mean(), d[1:].min(), float((d[1:] > d[0]).mean())))
for f in fills:
    A = np.array(res[f]); print(f"== {meth} fill={f} n={len(A)}: margin drop when overwriting the TRUE hole {A[:,2].mean():+.3f} (median {np.median(A[:,2]):+.3f}) | same shape on a watermarked area {A[:,3].mean():+.3f} (median {np.median(A[:,3]):+.3f})")
    print(f"     per image: share of moved shapes with a larger drop than the true hole = {A[:,5].mean():.3f}  (1.0 = perfect ranking) | AUC pooled over images, drop/margin: {auc(A[:,3]/A[:,1], A[:,2]/A[:,1]):.3f}")
    for lo, hi in [(0.003, .02), (.02, .06), (.06, .12), (.12, 1)]:
        k = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if k.sum(): print(f"     hole {lo:.3f}-{hi:.2f} n={k.sum():2d}: true {A[k,2].mean():+.3f}  moved {A[k,3].mean():+.3f}  ranking {A[k,5].mean():.3f}")
