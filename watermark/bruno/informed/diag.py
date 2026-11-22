"""Diagnose synthetic -> real transfer of an informed checkpoint: mean predicted probability inside the watermarked
area / inside clean holes, and mIoU, on (a) real full images, (b) synthetic composites (generic bases) at several
amplitudes, (c) in-domain synthetic composites (real-image bases watermarked with the kit embedder), same real masks."""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image, downsample_mask
from common import RELEASE
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=48)
a = ap.parse_args()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
wm = load_watermark(a.method, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
ds = LocalisationDataset(RELEASE, "train", methods=[a.method])
items = [ds[i] for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:a.n]
items = [it for it in items if (it["mask"] == 0).sum() > 1500]

@torch.no_grad()
def run(X, bits, masks):
    x = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).cuda().float(); b = torch.from_numpy(np.stack(bits)).cuda().float()
    ps = []
    for i in range(0, len(x), 16):
        t, _ = normalise(templ(x[i:i + 16], b[i:i + 16]))
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = net(x[i:i + 16], t)
        ps.append(torch.sigmoid(lg.float()).cpu().numpy())
    p = np.concatenate(ps); g = np.stack([downsample_mask(m.astype(np.float32)) for m in masks])
    return float(np.mean([pp[gg > 0].mean() for pp, gg in zip(p, g)])), float(np.mean([pp[gg == 0].mean() for pp, gg in zip(p, g)])), float(np.mean([miou_image(pp > 0.5, gg) for pp, gg in zip(p, g)]))

masks = [it["mask"].astype(bool) for it in items]
print(f"{a.method}  n={len(items)}   (mean prob in watermarked area | in clean holes | mIoU@0.5)")
print("  REAL full images                         : %.3f | %.3f | %.3f" % run([it["image"] for it in items], [it["message"] for it in items], masks))
clean = np.load(f"{SYN}/clean.npy", mmap_mode="r"); full = np.load(f"{SYN}/{a.method}_full_a.npy", mmap_mode="r"); msgs = np.load(f"{SYN}/{a.method}_msgs_a.npy")
for alpha in (1.5, 1.0, 0.8, 0.6):
    X = []
    for k, m in enumerate(masks):
        c = np.array(clean[300 + k]).astype(np.float32); w = np.array(full[300 + k]).astype(np.float32)
        X.append(np.clip(np.where(m[..., None], c + alpha * (w - c), c), 0, 255).round().astype(np.uint8))
    print("  synthetic generic bases, amplitude x%.1f   : %.3f | %.3f | %.3f" % ((alpha,) + run(X, [msgs[300 + k, 0] for k in range(len(masks))], masks)))
B = np.load(f"{SYN}/{a.method}_in_base.npy", mmap_mode="r"); F = np.load(f"{SYN}/{a.method}_in_full.npy", mmap_mode="r"); M = np.load(f"{out}/{a.method}_in_msgs.npy")
for alpha in (1.0, 0.8):
    X = []
    for k, m in enumerate(masks):
        c = np.array(B[k]).astype(np.float32); w = np.array(F[k, 0]).astype(np.float32)
        X.append(np.clip(np.where(m[..., None], c + alpha * (w - c), c), 0, 255).round().astype(np.uint8))
    print("  synthetic IN-DOMAIN bases, amplitude x%.1f  : %.3f | %.3f | %.3f" % ((alpha,) + run(X, [M[k, 0] for k in range(len(masks))], masks)))
# real images re-watermarked on top by us (same message): does the detector fire then?
X = [wm.embed(it["image"], it["message"]) for it in items]
print("  REAL full images, re-embedded by us       : %.3f | %.3f | %.3f" % run(X, [it["message"] for it in items], masks))
