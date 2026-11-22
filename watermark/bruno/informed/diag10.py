"""Frame hypothesis test with the informed detector: for REAL fully watermarked images with a weak response, search the
embedding frame (scale y/x, shift) that maximises the detector response; the template is computed on the hypothesised
frame (reflect padding outside the image) and cropped back to the visible image."""
import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
meth, ckpt = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
wm = load_watermark(meth, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
P = 64

def template_for_frame(x, b, sy, sx, dy, dx):
    """x: 1x3x512x512 [0,255]. Template in visible coordinates for the frame (scale sy, sx; centre shift dy, dx)."""
    xp = F.pad(x, (P, P, P, P), mode="reflect")
    H, W = int(round(512 * sy)), int(round(512 * sx)); y0 = P + 256 + dy - H // 2; x0 = P + 256 + dx - W // 2
    Tp = torch.zeros_like(xp); Tp[:, :, y0:y0 + H, x0:x0 + W] = templ(xp[:, :, y0:y0 + H, x0:x0 + W], b)
    return Tp[:, :, P:P + 512, P:P + 512]

@torch.no_grad()
def response(x, b, frames, g):
    out = []
    for i in range(0, len(frames), 16):
        T = torch.cat([template_for_frame(x, b, *f) for f in frames[i:i + 16]]); t, _ = normalise(T)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(x.expand(len(T), -1, -1, -1), t)[0].float())
        out += [(float(pp[g > 0].mean()), float(pp[g == 0].mean()) if (g == 0).any() else float("nan"), float(pp.mean())) for pp in p]
    return np.array(out)

ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rows = []
items = [ds[i] for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:40]
print(f"{meth}: detector prob (watermarked area | holes) with the visible frame -> with the best frame found   [frame = scale_y, scale_x, shift_y, shift_x]")
done = 0
for it in items:
    g = torch.from_numpy(downsample_mask(it["mask"].astype(np.float32))).cuda()
    x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].cuda().float(); b = torch.from_numpy(it["message"][None].astype(np.float32)).cuda()
    r0 = response(x, b, [(1.0, 1.0, 0, 0)], g)[0]
    if r0[0] > 0.6: continue
    c1 = [(s, s, dy, dx) for s in np.arange(0.92, 1.101, 0.02) for dy in (-8, 0, 8) for dx in (-8, 0, 8)]
    r1 = response(x, b, c1, g); k = int(r1[:, 2].argmax()); bst = c1[k]      # objective: mean prob over the whole image (no GT used)
    c2 = [(bst[0] + a, bst[1] + c, bst[2] + dy, bst[3] + dx) for a in (-0.02, -0.01, 0, 0.01, 0.02) for c in (-0.02, -0.01, 0, 0.01, 0.02) for dy in (-4, 0, 4) for dx in (-4, 0, 4)]
    r2 = response(x, b, c2, g); k2 = int(r2[:, 2].argmax())
    print("  %.2f | %.2f  ->  %.2f | %.2f   frame (%.2f, %.2f, %+d, %+d)" % (r0[0], r0[1], r2[k2, 0], r2[k2, 1], *c2[k2]), flush=True)
    rows.append((r0[0], r2[k2, 0], r0[1], r2[k2, 1])); done += 1
    if done >= N: break
R = np.array(rows); print("  mean: %.2f | %.2f  ->  %.2f | %.2f   (n=%d)   images reaching > 0.8: %.2f" % (R[:, 0].mean(), np.nanmean(R[:, 2]), R[:, 1].mean(), np.nanmean(R[:, 3]), len(R), np.mean(R[:, 1] > .8)))
