"""Estimate, per image, the frame on which the watermark was embedded (scale y, scale x, centre shift), by maximising
the decoder margin; frames larger than the image use reflect padding. REAL fully watermarked images vs a synthetic
control where the true frame is known (image = centre crop of a larger kit-embedded image, i.e. frame 1.06 x visible)."""
import os, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE, CLEAN  # noqa
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
meth = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
wm = load_watermark(meth, device="cuda")

def view(x, sy, sx, dy, dx, P=64):
    xp = F.pad(x, (P, P, P, P), mode="reflect")
    H, W = int(round(512 * sy)), int(round(512 * sx)); y0 = P + 256 + dy - H // 2; x0 = P + 256 + dx - W // 2
    return xp[:, :, y0:y0 + H, x0:x0 + W]

def marg(x, sm, cands):
    out = []
    for c in cands:
        lg = wm.logits(view(x, *c)).float().to(sm.device); out.append(float((lg[0] * sm).mean()))
    return np.array(out)

def estimate(img, msg):
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255).cuda(); sm = torch.from_numpy(msg.astype(np.float32) * 2 - 1).cuda()
    base = marg(x, sm, [(1, 1, 0, 0)])[0]
    c1 = [(s, s, dy, dx) for s in np.arange(0.92, 1.141, 0.02) for dy in (-10, 0, 10) for dx in (-10, 0, 10)]
    m1 = marg(x, sm, c1); b = c1[int(m1.argmax())]
    c2 = [(b[0] + a, b[1] + c, b[2] + dy, b[3] + dx) for a in (-0.02, -0.01, 0, 0.01, 0.02) for c in (-0.02, -0.01, 0, 0.01, 0.02) for dy in (-5, 0, 5) for dx in (-5, 0, 5)]
    c2 = [c for c in c2 if 0.9 <= c[0] <= 1.2 and 0.9 <= c[1] <= 1.2]
    m2 = marg(x, sm, c2); b2 = c2[int(m2.argmax())]
    return b2, float(m2.max() / max(base, 1e-6)), base

ds = LocalisationDataset(RELEASE, "train", methods=[meth])
real = [(ds[i]["image"], ds[i]["message"]) for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:N]
import glob
files = sorted(glob.glob(f"{CLEAN}/flux1024/**/*.jpg", recursive=True))[:6]; rng = np.random.default_rng(0); ctrl = []
for p in files:
    big = np.asarray(Image.open(p).convert("RGB").resize((544, 544), Image.Resampling.LANCZOS)); msg = rng.integers(0, 2, wm.nbits)
    w = wm.embed(big, msg); ctrl.append((w[16:528, 16:528], msg))        # visible 512 = centre crop of a 544 frame (scale 1.0625)
    w2 = wm.embed(big[16:528, 16:528], msg); ctrl.append((w2, msg))      # frame = visible (scale 1.0)
print(f"{meth}: estimated frame (scale_y, scale_x, shift_y, shift_x) | margin gain vs visible frame | margin at visible frame")
for k, (img, msg) in enumerate(ctrl):
    b, gain, base = estimate(img, msg); print("  CONTROL true scale %.4f : (%.2f, %.2f, %+d, %+d) | x%.2f | %.2f" % ((1.0625 if k % 2 == 0 else 1.0,) + tuple(b) + (gain, base)), flush=True)
est = []
for img, msg in real:
    b, gain, base = estimate(img, msg); est.append(b + (gain, base)); print("  REAL                    : (%.2f, %.2f, %+d, %+d) | x%.2f | %.2f" % (tuple(b) + (gain, base)), flush=True)
E = np.array(est); print("  REAL summary: scale_y median %.3f, scale_x median %.3f, share with both scales == 1.00: %.2f, mean gain x%.2f" % (np.median(E[:, 0]), np.median(E[:, 1]), np.mean((np.abs(E[:, 0] - 1) < .005) & (np.abs(E[:, 1] - 1) < .005)), E[:, 4].mean()))
