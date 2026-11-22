"""Plain sliding-patch occlusion on the real full training images.
usage: occ.py <method> <tag> <cfg> [...]  with cfg = patch:stride:fill   e.g. 32:16:mean
Evidence of a 4x4 cell = mean over the patches covering it of (margin(image) - margin(image with patch filled))."""
import sys, time
from lib import *

meth, tag = sys.argv[1], sys.argv[2]
cfgs = sys.argv[3:]
NIMG = int(os.environ.get("NIMG", 0))
set_log(f"occ_{meth}_{tag}"); wm = load_wm(meth); G = gt128()
items = [it for it in load_items("train", meth) if it["full"]]
if NIMG:
    items = items[:NIMG]
part = [it for it in load_items("train", meth) if not it["full"]]
names = [it["name"] for it in items]
save = dict(names=np.array(names))
for cfg in cfgs:
    s, st, fk = cfg.split(":"); s = int(s); st = int(st)
    pos = [(y, x) for y in range(0, 512 - s + 1, st) for x in range(0, 512 - s + 1, st)]
    cov = torch.zeros(len(pos), 128, 128, device=DEV)            # coverage of each patch on the 128 grid
    for k, (y, x) in enumerate(pos):
        cov[k, y // 4:(y + s) // 4, x // 4:(x + s) // 4] = 1
    cnt = cov.sum(0).clamp(min=1)
    ev = {k: [] for k in ["mean", "tstat"]}
    t0 = time.time()
    for j, it in enumerate(items):
        img = load_img(it["path"]); sm = sm_of(it["bits"])
        fill = make_fill(img, fk, load_img(part[j % len(part)]["path"]) if fk == "other" else None)
        y0 = slog(wm, img, sm)                                    # 1 x nbits
        D = []
        for i in range(0, len(pos), 128):
            xb = img.repeat(len(pos[i:i + 128]), 1, 1, 1)
            for b, (y, x) in enumerate(pos[i:i + 128]):
                xb[b, :, y:y + s, x:x + s] = fill[0, :, y:y + s, x:x + s]
            D.append(y0 - slog(wm, xb, sm))
        D = torch.cat(D)                                          # npos x nbits drop of each signed logit
        ev["mean"].append((torch.einsum("k,khw->hw", D.mean(1), cov) / cnt).cpu().numpy())
        ev["tstat"].append((torch.einsum("k,khw->hw", D.mean(1) / (D.std(1) + 1e-6), cov) / cnt).cpu().numpy())
    dt = (time.time() - t0) / len(items)
    print(f"== {meth} occlusion {cfg}: {len(pos)} queries, {dt:.2f} s/image", flush=True)
    for k, v in ev.items():
        v = np.array(v); save[f"{cfg}|{k}"] = v.astype(np.float16)
        for sm_ in [0.0, 2.0]:
            evaluate(to128(v, smooth=sm_), names, G, f"occ {cfg} {k} sm{sm_}")
np.savez(f"{OUT}/occl_raw_{meth}_{tag}.npz", **save)
