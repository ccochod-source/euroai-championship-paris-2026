"""Does the detector rely on the kit's uint8 quantisation (TrustMark / StegaStamp truncate, PixelSeal rounds)?
In-domain bases, real masks; watermarked image built as q(C + r) with r the float residual and q = floor / round /
ceil / stochastic rounding; plus the kit's own output as control."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
meth, ckpt = sys.argv[1], sys.argv[2]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
wm = load_watermark(meth, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
ds = LocalisationDataset(RELEASE, "train", methods=[meth])
masks = [ds[i]["mask"].astype(bool) for i, m in enumerate(ds.meta) if m.get("full_watermark", False)]
masks = [m for m in masks if (~m).sum() > 1500][:32]; g = np.stack([downsample_mask(m.astype(np.float32)) for m in masks])
B = np.load(f"{SYN}/{meth}_in_base.npy", mmap_mode="r"); M = np.load(f"{out}/{meth}_in_msgs.npy"); Fu = np.load(f"{SYN}/{meth}_in_full.npy", mmap_mode="r")
N = len(masks); C = np.stack([np.array(B[k]) for k in range(N)]); bits = np.stack([M[k, 0] for k in range(N)])
with torch.no_grad():
    r = torch.cat([templ(torch.from_numpy(C[i:i + 8]).permute(0, 3, 1, 2).cuda().float(), torch.from_numpy(bits[i:i + 8]).cuda()) for i in range(0, N, 8)]).permute(0, 2, 3, 1).cpu().numpy()
rng = np.random.default_rng(0)
Q = {"kit output": None, "floor(C + r)": np.floor, "round(C + r)": np.round, "ceil(C + r)": np.ceil, "stochastic round": lambda v: np.floor(v + rng.random(v.shape))}

@torch.no_grad()
def run(X):
    x = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).cuda().float(); b = torch.from_numpy(bits).cuda().float(); ps = []
    for i in range(0, len(x), 8):
        t, _ = normalise(templ(x[i:i + 8], b[i:i + 8]))
        with torch.autocast("cuda", dtype=torch.float16):
            ps.append(torch.sigmoid(net(x[i:i + 8], t)[0].float()).cpu().numpy())
    p = np.concatenate(ps)
    return float(np.mean([pp[gg > 0].mean() for pp, gg in zip(p, g)])), float(np.mean([pp[gg == 0].mean() for pp, gg in zip(p, g)]))

print(meth, " (detector prob in watermarked area | in clean holes)")
for name, q in Q.items():
    X = []
    for k in range(N):
        w = np.array(Fu[k, 0]) if q is None else np.clip(q(C[k].astype(np.float32) + r[k]), 0, 255).astype(np.uint8)
        X.append(np.where(masks[k][..., None], w, C[k]))
    d = (np.stack(X).astype(np.float32) - C)[np.stack(masks)]
    print("  %-18s %.3f | %.3f    residual mean %+.3f, mean abs %.3f" % ((name,) + run(X) + (d.mean(), np.abs(d).mean())))
