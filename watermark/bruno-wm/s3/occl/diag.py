"""Diagnostics of the random-mask regression: is the weak AUC due to estimation noise (too few queries, non-linear
interactions) or to a real heterogeneity of the watermark contribution? usage: diag.py <method> <g> <K> <p> <fill> [n]"""
import sys
from lib import *

meth = sys.argv[1]; g = int(sys.argv[2]); K = int(sys.argv[3]); p = float(sys.argv[4]); fk = sys.argv[5]
n = int(sys.argv[6]) if len(sys.argv) > 6 else 16
set_log(f"diag_{meth}"); wm = load_wm(meth); G = gt128()
items = [it for it in load_items("train", meth) if it["full"]][:n]
cells, masks = rand_masks(K, g, p, False, seed=7); masks = masks.half()
print(f"== diag {meth} g={g} K={K} p={p} fill={fk}")
rows = []
for it in items:
    img = load_img(it["path"]); sm = sm_of(it["bits"]); fill = make_fill(img, fk)
    gt = G[it["name"]]
    frac = F.adaptive_avg_pool2d(torch.from_numpy(gt.astype(np.float32))[None, None], g)[0, 0].numpy().ravel()  # wm fraction per cell
    Y = torch.cat([slog(wm, (m.float() * img + (1 - m.float()) * fill), sm) for m in masks.split(128)])
    y = Y.mean(1, keepdim=True); y0 = slog(wm, img, sm).mean().item()
    out = [f"{it['name'][:6]} hole {1-gt.mean():.3f} m0 {y0:+.2f} mean(y) {y.mean().item():+.2f}"]
    for k in [K // 8, K // 4, K // 2, K]:
        w = ridge(cells[:k], y[:k], 0.05)
        res = (y[:k] - y[:k].mean()) - (cells[:k] - cells[:k].mean(0)) @ w
        r2 = 1 - res.var().item() / y[:k].var().item()
        wn = w[:, 0].cpu().numpy()
        a = auc_img(to128(wn.reshape(1, g, g))[0], gt)
        out.append(f"K{k}: R2 {r2:.2f} auc {a:.2f}")
    wa = ridge(cells[:K // 2], y[:K // 2], 0.05)[:, 0]; wb = ridge(cells[K // 2:], y[K // 2:], 0.05)[:, 0]
    sh = torch.corrcoef(torch.stack([wa, wb]))[0, 1].item()
    hw = wn[frac < 0.05]; ww = wn[frac > 0.95]
    out.append(f"split-half r {sh:.2f} | w hole cells {hw.mean() if len(hw) else float('nan'):+.4f} (n{len(hw)}) "
               f"wm cells {ww.mean():+.4f} +- {ww.std():.4f} (5%/50%/95%: {np.percentile(ww,5):+.4f} {np.percentile(ww,50):+.4f} {np.percentile(ww,95):+.4f})")
    print(" | ".join(out))
