"""Gradient-based attribution through the torch decoders (trustmark, pixelseal) on the real full training images.
usage: grad.py <method> <tag>
Maps: grad x (image - baseline) and integrated gradients (32 steps) for baselines blur16 / mean / grey, summed over
channels and 4x4 cells -> 128x128."""
import sys, time
from lib import *

meth, tag = sys.argv[1], sys.argv[2]
NIMG = int(os.environ.get("NIMG", 0)); STEPS = 32
set_log(f"grad_{meth}_{tag}"); wm = load_wm(meth); G = gt128()
items = [it for it in load_items("train", meth) if it["full"]]
if NIMG:
    items = items[:NIMG]
names = [it["name"] for it in items]
bases = ["blur16", "mean", "grey", "blur4"]
ev = {}
t0 = time.time()
for it in items:
    img = load_img(it["path"]); sm = sm_of(it["bits"])
    x = img.clone().requires_grad_(True)
    with torch.enable_grad():
        g0 = torch.autograd.grad(slog_grad(wm, x, sm).mean(), x)[0]
    ev.setdefault("gradabs", []).append(F.avg_pool2d(g0.abs().sum(1, keepdim=True), 4)[0, 0].cpu().numpy())
    for bk in bases:
        base = make_fill(img, bk)
        ev.setdefault(f"gxi_{bk}", []).append(F.avg_pool2d((g0 * (img - base)).sum(1, keepdim=True), 4)[0, 0].cpu().numpy())
        al = ((torch.arange(STEPS, device=DEV) + 0.5) / STEPS).view(-1, 1, 1, 1)
        acc = torch.zeros_like(img)
        for a in al.split(16):
            xa = (base + a * (img - base)).requires_grad_(True)
            with torch.enable_grad():
                acc += torch.autograd.grad(slog_grad(wm, xa, sm).mean(1).sum(), xa)[0].sum(0, keepdim=True)
        ig = acc / STEPS * (img - base)
        ev.setdefault(f"ig_{bk}", []).append(F.avg_pool2d(ig.sum(1, keepdim=True), 4)[0, 0].cpu().numpy())
print(f"== {meth} gradients: {(time.time()-t0)/len(items):.2f} s/image for all {2*len(bases)+1} maps, "
      f"mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
save = dict(names=np.array(names))
for k, v in ev.items():
    v = np.array(v); save[k] = v.astype(np.float32)
    for sm_ in [0.0, 2.0, 4.0]:
        evaluate(to128(v, smooth=sm_), names, G, f"{k} sm{sm_}")
np.savez(f"{OUT}/occl_raw_{meth}_{tag}.npz", **save)
