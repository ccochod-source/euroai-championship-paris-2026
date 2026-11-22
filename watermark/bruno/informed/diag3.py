"""Which post-processing of a kit-embedded composite reproduces the detector's weak response on REAL images?
In-domain bases + real masks; candidates applied to the composite X = where(mask, embed(C, m), C)."""
import argparse, io, os, sys, numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask, to_tensor
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=32)
a = ap.parse_args()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
wm = load_watermark(a.method, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
ds = LocalisationDataset(RELEASE, "train", methods=[a.method])
items = [ds[i] for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:a.n]
items = [it for it in items if (it["mask"] == 0).sum() > 1500]
masks = [it["mask"].astype(bool) for it in items]; g = np.stack([downsample_mask(m.astype(np.float32)) for m in masks])
B = np.load(f"{SYN}/{a.method}_in_base.npy", mmap_mode="r"); M = np.load(f"{out}/{a.method}_in_msgs.npy"); Fu = np.load(f"{SYN}/{a.method}_in_full.npy", mmap_mode="r")
rng = np.random.default_rng(0)

def jpeg(x, q):
    b = io.BytesIO(); Image.fromarray(x).save(b, "JPEG", quality=q, subsampling=0); return np.asarray(Image.open(b).convert("RGB"))
def rsz(x, s, how=Image.Resampling.LANCZOS): return np.asarray(Image.fromarray(x).resize((s, s), how))

def make(k, mode):
    c = np.array(B[k]); m = masks[k]; msg = M[k, 0]
    if mode == "up2x":                      # embed and composite at 1024, then downscale
        c2 = rsz(c, 1024); w2 = wm.embed(c2, msg); m2 = cv2.resize(m.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST) > 0
        return rsz(np.where(m2[..., None], w2, c2), 512)
    w = np.array(Fu[k, 0]); x = np.where(m[..., None], w, c)
    if mode == "direct": return x
    if mode.startswith("jpeg"): return jpeg(x, int(mode[4:]))
    if mode == "resize_roundtrip": return rsz(rsz(x, 1024), 512)
    if mode == "resize_768": return rsz(rsz(x, 768), 512)
    if mode == "noise1": return np.clip(x + rng.normal(0, 1, x.shape), 0, 255).round().astype(np.uint8)
    if mode == "noise2": return np.clip(x + rng.normal(0, 2, x.shape), 0, 255).round().astype(np.uint8)
    if mode == "blur0.5": return cv2.GaussianBlur(x, (0, 0), 0.5)
    if mode == "wm_jpeg90_then_composite": return np.where(m[..., None], jpeg(w, 90), c)
    if mode == "shift1px":
        w1 = np.roll(wm.embed(np.roll(c, 1, 1), msg), -1, 1); return np.where(m[..., None], w1, c)

@torch.no_grad()
def run(X, bits):
    x = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).cuda().float(); b = torch.from_numpy(np.stack(bits)).cuda().float(); ps = []
    for i in range(0, len(x), 8):
        t, _ = normalise(templ(x[i:i + 8], b[i:i + 8]))
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = net(x[i:i + 8], t)
        ps.append(torch.sigmoid(lg.float()).cpu().numpy())
    p = np.concatenate(ps)
    lgs = [wm.logits(to_tensor(xx))[0].float().cpu().numpy() for xx in X]
    mar = float(np.mean([(l * (bb * 2.0 - 1)).mean() for l, bb in zip(lgs, bits)])); acc = float(np.mean([((l > 0) == (bb > 0)).mean() for l, bb in zip(lgs, bits)]))
    return float(np.mean([pp[gg > 0].mean() for pp, gg in zip(p, g)])), float(np.mean([pp[gg == 0].mean() for pp, gg in zip(p, g)])), mar, acc

print(a.method, "n =", len(items), "  (detector prob in watermarked area | in clean holes | decoder margin | bit acc)")
print("  %-26s %.3f | %.3f | %5.2f | %.3f" % (("REAL images",) + run([it["image"] for it in items], [it["message"] for it in items])))
for mode in ["direct", "up2x", "resize_roundtrip", "resize_768", "jpeg98", "jpeg95", "jpeg90", "jpeg80", "wm_jpeg90_then_composite", "noise1", "noise2", "blur0.5", "shift1px"]:
    print("  %-26s %.3f | %.3f | %5.2f | %.3f" % ((mode,) + run([make(k, mode) for k in range(len(items))], [M[k, 0] for k in range(len(items))])))
