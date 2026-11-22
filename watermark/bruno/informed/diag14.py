"""Hypothesis: in the real images the clean region does NOT contain the original pixels the embedder saw (an object was
added / removed there), so re-embedding the visible image gives a different residual everywhere.
Synthetic test on in-domain bases with real masks: W = embed(C, m) ; hole filled with (a) C itself, (b) pixels of
another image, (c) an inpainted version of C, (d) a blurred C, (e) grey. Detector response with the standard template,
and with a template computed after inpainting the (true) hole."""
import os, sys, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
meth = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
wm = load_watermark(meth, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(f"out/{meth}_i1_last.pt", map_location="cpu", weights_only=True))
ds = LocalisationDataset(RELEASE, "train", methods=[meth])
masks = [ds[i]["mask"].astype(bool) for i, m in enumerate(ds.meta) if m.get("full_watermark", False)]
masks = [m for m in masks if (~m).sum() > 1500][:40]; g = np.stack([downsample_mask(m.astype(np.float32)) for m in masks])
B = np.load(f"{SYN}/{meth}_in_base.npy", mmap_mode="r"); M = np.load(f"{out}/{meth}_in_msgs.npy"); Fu = np.load(f"{SYN}/{meth}_in_full.npy", mmap_mode="r")
N = len(masks); bits = np.stack([M[k, 0] for k in range(N)]); holefrac = np.array([1 - m.mean() for m in masks])

def fill(k, mode):
    c = np.array(B[k]); w = np.array(Fu[k, 0]); m = masks[k]; hole = (~m).astype(np.uint8)
    if mode == "original pixels": d = c
    elif mode == "other image": d = np.array(B[(k + 97) % len(B)])
    elif mode == "inpainted": d = cv2.inpaint(c, cv2.dilate(hole, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
    elif mode == "blurred": d = cv2.GaussianBlur(c, (0, 0), 6)
    elif mode == "grey": d = np.full_like(c, 128)
    return np.where(m[..., None], w, d)

@torch.no_grad()
def run(X, Xt=None):
    x = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).cuda().float(); b = torch.from_numpy(bits).cuda().float(); ps = []
    xt = x if Xt is None else torch.from_numpy(np.stack(Xt)).permute(0, 3, 1, 2).cuda().float()
    for i in range(0, len(x), 8):
        t, _ = normalise(templ(xt[i:i + 8], b[i:i + 8]))
        with torch.autocast("cuda", dtype=torch.float16):
            ps.append(torch.sigmoid(net(x[i:i + 8], t)[0].float()).cpu().numpy())
    p = np.concatenate(ps); w = np.array([pp[gg > 0].mean() for pp, gg in zip(p, g)]); h = np.array([pp[gg == 0].mean() for pp, gg in zip(p, g)])
    return w, h

small, large = holefrac < np.median(holefrac), holefrac >= np.median(holefrac)
print(f"{meth} n={N}  detector prob: watermarked area (small holes | large holes) | in holes  -- hole content:")
for mode in ["original pixels", "other image", "inpainted", "blurred", "grey"]:
    X = [fill(k, mode) for k in range(N)]; w, h = run(X)
    print("  %-16s %.3f (%.3f | %.3f) | %.3f     share of images > 0.8: %.2f, < 0.4: %.2f" % (mode, w.mean(), w[small].mean(), w[large].mean(), h.mean(), np.mean(w > .8), np.mean(w < .4)))
    if mode == "other image":
        Xi = [np.where(masks[k][..., None], X[k], cv2.inpaint(X[k], cv2.dilate((~masks[k]).astype(np.uint8), np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)) for k in range(N)]
        w, h = run(X, Xi); print("    + template after inpainting the true hole: %.3f (%.3f | %.3f) | %.3f" % (w.mean(), w[small].mean(), w[large].mean(), h.mean()))
        Xo = [np.where(masks[k][..., None], X[k], np.array(B[k])) for k in range(N)]
        w, h = run(X, Xo); print("    + template with the ORIGINAL pixels in the hole (oracle): %.3f (%.3f | %.3f) | %.3f" % (w.mean(), w[small].mean(), w[large].mean(), h.mean()))
