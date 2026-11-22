"""Hypothesis: the real images were watermarked at a HIGHER resolution (true 1024 px originals) and downscaled to 512.
True FLUX 1024 originals: embed at 1024 (kit), composite with a real mask at 1024, downscale to 512 with several kernels,
then run the informed detector (standard template from the 512 image). Control: embed directly at 512."""
import argparse, glob, os, sys, numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE, CLEAN  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask, to_tensor
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=24)
a = ap.parse_args()
wm = load_watermark(a.method, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
files = sorted(glob.glob(f"{CLEAN}/flux1024/**/*.jpg", recursive=True) + glob.glob(f"{CLEAN}/flux1024/**/*.png", recursive=True))
big = []
for p in files:
    im = np.asarray(Image.open(p).convert("RGB"))
    if im.shape[0] >= 1024 and im.shape[1] >= 1024: big.append(im[:1024, :1024])
    if len(big) >= a.n: break
print("true 1024 originals:", len(big), "of", len(files), "files; first shape", np.asarray(Image.open(files[0])).shape)
ds = LocalisationDataset(RELEASE, "train", methods=[a.method])
masks = [ds[i]["mask"].astype(bool) for i, m in enumerate(ds.meta) if m.get("full_watermark", False)]
masks = [m for m in masks if (~m).sum() > 1500][:len(big)]
g = np.stack([downsample_mask(m.astype(np.float32)) for m in masks]); rng = np.random.default_rng(0)
K = {"lanczos": Image.Resampling.LANCZOS, "bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR, "box": Image.Resampling.BOX}
def down(x, how): return np.asarray(Image.fromarray(x).resize((512, 512), K[how]))

@torch.no_grad()
def run(X, bits):
    x = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).cuda().float(); b = torch.from_numpy(np.stack(bits)).cuda().float(); ps = []
    for i in range(0, len(x), 8):
        t, _ = normalise(templ(x[i:i + 8], b[i:i + 8]))
        with torch.autocast("cuda", dtype=torch.float16):
            ps.append(torch.sigmoid(net(x[i:i + 8], t)[0].float()).cpu().numpy())
    p = np.concatenate(ps); lg = [wm.logits(to_tensor(xx))[0].float().cpu().numpy() for xx in X]
    return (float(np.mean([pp[gg > 0].mean() for pp, gg in zip(p, g)])), float(np.mean([pp[gg == 0].mean() for pp, gg in zip(p, g)])),
            float(np.mean([(l * (bb * 2.0 - 1)).mean() for l, bb in zip(lg, bits)])), float(np.mean([((l > 0) == (bb > 0)).mean() for l, bb in zip(lg, bits)])))

bits = [rng.integers(0, 2, wm.nbits) for _ in big]
W = [wm.embed(B, b) for B, b in zip(big, bits)]
print(a.method, " (detector prob in watermarked area | clean holes | decoder margin | bit acc)   REAL reference: stegastamp 0.44|0.13|6.18  trustmark 0.38|0.21|7.57")
for how in K:
    X = []
    for B, w, m in zip(big, W, masks):
        m2 = cv2.resize(m.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST) > 0
        X.append(down(np.where(m2[..., None], w, B), how))
    print("  embed at 1024, composite, downscale %-9s: %.3f | %.3f | %5.2f | %.3f" % ((how,) + run(X, bits)))
X = []
for B, b, m in zip(big, bits, masks):
    c = down(B, "lanczos"); X.append(np.where(m[..., None], wm.embed(c, b), c))
print("  control: downscale lanczos then embed at 512 : %.3f | %.3f | %5.2f | %.3f" % run(X, bits))
