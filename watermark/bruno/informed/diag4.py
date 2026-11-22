"""Does the kit embedder explain the REAL watermark when the frame is known exactly?
Real not-full images whose mask is a plain rectangle: crop the rectangle (the watermarked original), upscale to 512 and
run the informed detector with the template of the upscaled crop. Control: synthetic rectangles embedded by the kit
(crop embedded with a known message), same processing. Prints mean detector probability and decoder accuracy."""
import argparse, os, sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True)
a = ap.parse_args()
wm = load_watermark(a.method, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
up = lambda c: np.asarray(Image.fromarray(np.ascontiguousarray(c)).resize((512, 512), Image.Resampling.BICUBIC))

@torch.no_grad()
def run(crops, bits):
    out = []
    for c, b in zip(crops, bits):
        acc = float(((wm.logits(to_tensor(np.ascontiguousarray(c)))[0].float().cpu().numpy() > 0) == (b > 0)).mean())
        x = torch.from_numpy(up(c)).permute(2, 0, 1)[None].cuda().float(); bb = torch.from_numpy(b[None].astype(np.float32)).cuda()
        t, _ = normalise(templ(x, bb)); tw, _ = normalise(templ(x, 1 - bb))
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(x, t)[0].float()).mean().item(); pw = torch.sigmoid(net(x, tw)[0].float()).mean().item()
        out.append((p, pw, acc, min(c.shape[:2])))
    return np.array(out)

ds = LocalisationDataset(RELEASE, "train", methods=[a.method]); crops, bits = [], []
for i, m in enumerate(ds.meta):
    if m.get("full_watermark", False): continue
    it = ds[i]; mk = it["mask"]; ys, xs = np.nonzero(mk)
    if len(ys) == 0: continue
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if mk[y0:y1, x0:x1].mean() < 0.98 or min(y1 - y0, x1 - x0) < 90 or max(y1 - y0, x1 - x0) / min(y1 - y0, x1 - x0) > 2: continue
    crops.append(it["image"][y0:y1, x0:x1]); bits.append(it["message"])
R = run(crops, bits)
rect = np.load(f"{SYN}/{a.method}_rect_a.npy", mmap_mode="r"); rects = np.load(f"{SYN}/{a.method}_rects_a.npy"); msgs = np.load(f"{SYN}/{a.method}_msgs_a.npy")
sc, sb = [], []
for j in range(3000):
    y0, y1, x0, x1 = rects[j]; h, w = y1 - y0, x1 - x0
    if h <= 0 or min(h, w) < 90 or max(h, w) / min(h, w) > 2: continue
    sc.append(np.array(rect[j][y0:y1, x0:x1])); sb.append(msgs[j, 1])
    if len(sc) >= max(12, len(crops)): break
S = run(sc, sb)
print(f"{a.method}: detector on the watermarked rectangle upscaled to 512  (mean prob, true message | flipped message | decoder acc | n)")
print("  REAL plain rectangles      : %.3f | %.3f | %.3f | n=%d   (min side median %d px)" % (R[:, 0].mean(), R[:, 1].mean(), R[:, 2].mean(), len(R), np.median(R[:, 3])))
print("  SYNTHETIC kit rectangles   : %.3f | %.3f | %.3f | n=%d   (min side median %d px)" % (S[:, 0].mean(), S[:, 1].mean(), S[:, 2].mean(), len(S), np.median(S[:, 3])))
