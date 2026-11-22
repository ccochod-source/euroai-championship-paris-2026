"""Random-mask regression (RISE / LIME style) on the real full training images.
usage: rise.py <method> <tag> <cfg> [<cfg> ...]   with cfg = g:K:p:fill:soft(0/1)   e.g. 16:1024:0.8:blur8:0
Saves OUT/occl_raw_<method>_<tag>.npz with the g x g maps of every config and prints AUC / cross-validated mIoU."""
import sys, time
from lib import *

meth, tag = sys.argv[1], sys.argv[2]
cfgs = sys.argv[3:]
NIMG = int(os.environ.get("NIMG", 0))
set_log(f"rise_{meth}_{tag}"); wm = load_wm(meth); G = gt128()
items = [it for it in load_items("train", meth) if it["full"]]
if NIMG:
    items = items[:NIMG]
part = [it for it in load_items("train", meth) if not it["full"]]
names = [it["name"] for it in items]
save = dict(names=np.array(names))
for cfg in cfgs:
    g, K, p, fk, soft = cfg.split(":"); g = int(g); K = int(K); p = float(p); soft = bool(int(soft))
    cells, masks = rand_masks(K, g, p, soft, seed=g * 1000 + K)
    masks = masks.half()
    ev = {k: [] for k in ["mean", "tstat", "frac", "sat"]}
    t0 = time.time()
    for j, it in enumerate(items):
        img = load_img(it["path"]); sm = sm_of(it["bits"])
        fill = make_fill(img, fk, load_img(part[j % len(part)]["path"]) if fk == "other" else None)
        Y = torch.cat([slog(wm, (m.float() * img + (1 - m.float()) * fill), sm) for m in masks.split(128)])  # K x nbits
        Wb = ridge(cells, Y, 0.05)                                   # P x nbits
        ev["mean"].append(Wb.mean(1).view(g, g).cpu().numpy())
        ev["tstat"].append((Wb.mean(1) / (Wb.std(1) + 1e-6)).view(g, g).cpu().numpy())
        ev["frac"].append((Wb > 0).float().mean(1).view(g, g).cpu().numpy())
        ysat = torch.tanh(Y / 4).mean(1, keepdim=True)              # saturating per-bit response
        ev["sat"].append(ridge(cells, ysat, 0.05)[:, 0].view(g, g).cpu().numpy())
    dt = (time.time() - t0) / len(items)
    print(f"== {meth} {cfg}: {dt:.2f} s/image", flush=True)
    for k, v in ev.items():
        v = np.array(v); save[f"{cfg}|{k}"] = v.astype(np.float32)
        for sm_ in ([0.0, 2.0] if k in ("mean", "tstat") else [0.0]):
            evaluate(to128(v, smooth=sm_), names, G, f"{cfg} {k} sm{sm_}")
np.savez(f"{OUT}/occl_raw_{meth}_{tag}.npz", **save)
