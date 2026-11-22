"""Sanity: which fills kill the watermark (margin of the fully filled image), throughput, linearity of the margin
in the kept fraction. usage: exp0.py <method> [n images]"""
import sys, time
from lib import *

meth = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
wm = load_wm(meth); G = gt128()
items = [it for it in load_items("train", meth) if it["full"]][:n]
part = [it for it in load_items("train", meth) if not it["full"]]
kinds = ["grey", "mean", "blur1", "blur2", "blur4", "blur8", "blur16", "down2", "down4", "down8", "shift2", "shift8", "flip", "other"]
rows = []
for j, it in enumerate(items):
    img = load_img(it["path"]); sm = sm_of(it["bits"])
    oth = load_img(part[j]["path"])
    xs = torch.cat([img] + [make_fill(img, k, oth) for k in kinds])
    rows.append(slog(wm, xs, sm).mean(1).cpu().numpy())
rows = np.array(rows)
print("margin of the whole image after replacing EVERYTHING by the fill (mean over %d full images)" % len(items))
for k, v in zip(["image"] + kinds, rows.mean(0)):
    print(f"  {k:8s} {v:+.3f}")

# throughput + margin versus kept fraction (random 16x16 cells, blur8 fill)
it = items[0]; img = load_img(it["path"]); sm = sm_of(it["bits"]); fill = make_fill(img, "blur8")
for p in [0.95, 0.9, 0.8, 0.7, 0.5, 0.3]:
    cells, m = rand_masks(256, 16, p, False, seed=1)
    y = query_masks(wm, img, fill, m, sm).mean(1)
    print(f"keep p={p}: margin mean {y.mean():+.2f} std {y.std():.2f}")
torch.cuda.synchronize(); t = time.time()
cells, m = rand_masks(1024, 16, 0.8, False, seed=2); y = query_masks(wm, img, fill, m, sm); torch.cuda.synchronize()
print(f"throughput: {1024/(time.time()-t):.0f} queries/s, mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
