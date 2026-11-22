"""Which template makes the (synthetic-trained) informed detector fire on REAL fully watermarked images?
Templates: standard (re-embed X at 512), sign-flipped, from X upsampled x2 (then residual downscaled), from X at 256
(residual upsampled), from blurred X, from a 4-px inward crop, with a wrong message. Prints prob in watermarked area /
clean holes. If one alternative restores ~1.0 / ~0.1 the real embedding pipeline is identified."""
import argparse, os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=32)
a = ap.parse_args()
wm = load_watermark(a.method, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
ds = LocalisationDataset(RELEASE, "train", methods=[a.method])
items = [ds[i] for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:a.n]
items = [it for it in items if (it["mask"] == 0).sum() > 1500]
x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).cuda().float()
b = torch.from_numpy(np.stack([it["message"] for it in items])).cuda().float()
g = np.stack([downsample_mask(it["mask"].astype(np.float32)) for it in items])
rs = lambda t, s: F.interpolate(t, size=(s, s), mode="bicubic", align_corners=False, antialias=True)

def T_std(x, b): return templ(x, b)
def T_up(x, b): return rs(templ(rs(x, 1024).clamp(0, 255), b), 512)
def T_down(x, b): return rs(templ(rs(x, 256).clamp(0, 255), b), 512)
def T_768(x, b): return rs(templ(rs(x, 768).clamp(0, 255), b), 512)
def T_blur(x, b): return templ(rs(rs(x, 384), 512).clamp(0, 255), b)
def T_crop(x, b):
    t = torch.zeros_like(x); t[:, :, 4:-4, 4:-4] = templ(x[:, :, 4:-4, 4:-4], b); return t
def T_wrong(x, b): return templ(x, 1 - b)
def T_shuf(x, b): return templ(x, b.roll(1, 0))

@torch.no_grad()
def run(fn, sign=1.0):
    ps = []
    for i in range(0, len(x), 8):
        t, _ = normalise(sign * fn(x[i:i + 8], b[i:i + 8]))
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = net(x[i:i + 8], t)
        ps.append(torch.sigmoid(lg.float()).cpu().numpy())
    p = np.concatenate(ps)
    return float(np.mean([pp[gg > 0].mean() for pp, gg in zip(p, g)])), float(np.mean([pp[gg == 0].mean() for pp, gg in zip(p, g)]))

print(a.method, "n =", len(items), "  (prob in watermarked area | prob in clean holes) on REAL full images")
for name, fn, sg in [("standard 512", T_std, 1), ("sign flipped", T_std, -1), ("embed at 1024", T_up, 1), ("embed at 768", T_768, 1), ("embed at 256", T_down, 1),
                     ("from blurred X", T_blur, 1), ("4px inward crop", T_crop, 1), ("flipped bits", T_wrong, 1), ("other image's message", T_shuf, 1)]:
    print("  %-22s %.3f | %.3f" % ((name,) + run(fn, sg)))
