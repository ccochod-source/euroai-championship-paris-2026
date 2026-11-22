"""Hypothesis: the watermarked image passed through a generative model's autoencoder. Fingerprints after a VAE
round trip of our embedding, to compare with the challenge images."""
import sys, numpy as np, torch
from common import OUT, RELEASE, W
from model import Net
from vae import load_vae
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
from bands import stat
TARGET = {"stegastamp": "margin 6.08 | slopes +0.040 +0.306 +0.385 +0.401 +0.452 | det0 0.110 det1 0.905",
          "trustmark": "margin 7.83 | slopes +0.037 +0.100 +0.088 +0.069 +0.092", "pixelseal": "margin 8.96 | slopes -0.016 -0.015 +0.008 +0.032 +0.409"}
nets = []
for ck in (f"{OUT}/stegastamp_v1_last.pt", f"{OUT}/stegastamp_ftprobe_fold0.pt"):
    n_ = Net().cuda().eval(); n_.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); nets.append(n_)
vaes = {k: load_vae(f"{W}/vae/{k}").cuda() for k in sys.argv[1:]}
def rt(v, a): return v.roundtrip(torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].cuda())[0].permute(1, 2, 0).round().byte().cpu().numpy()
@torch.no_grad()
def det(a):
    x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda()
    with torch.autocast("cuda", dtype=torch.float16): return [float(torch.sigmoid(n_(x).float()).mean()) for n_ in nets]
full = np.ones((512, 512), bool)
for meth in ("stegastamp", "trustmark", "pixelseal"):
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
    bases = [ds[i]["image"] for i in range(len(ds)) if not ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() < 0.05][:16]
    print(f"== {meth}: REAL challenge images: {TARGET[meth]}")
    cands = {"our embedding": lambda w, c: w}
    for k, v in vaes.items():
        cands[f"VAE {k} (embed, then VAE)"] = (lambda w, c, v=v: rt(v, w))
        cands[f"VAE {k} twice"] = (lambda w, c, v=v: rt(v, rt(v, w)))
        cands[f"VAE {k} on cover, then embed"] = (lambda w, c, v=v: None)
    for name, fn in cands.items():
        rows = []
        for c in bases:
            msg = rng.integers(0, 2, wm.nbits); s = 2.0 * msg - 1
            if name.endswith("then embed"):
                k = name.split()[1]; x = wm.embed(rt(vaes[k], c), msg)
            else: x = fn(wm.embed(c, msg), c)
            lg = wm.logits(to_tensor(x))[0].cpu().numpy()
            T = wm.embed(x, msg).astype(np.float32) - wm.embed(x, 1 - msg).astype(np.float32)
            rows.append([float((lg * s).mean())] + [v_ for v_, _ in stat(x.astype(np.float32), T, full)] + (det(x) if meth == "stegastamp" else []))
        m = np.mean(rows, 0)
        print(f"   {name:34s} margin {m[0]:5.2f} | slopes " + " ".join("%+.3f" % t for t in m[1:6]) + (" | det0 %.3f det1 %.3f" % (m[6], m[7]) if meth == "stegastamp" else ""), flush=True)
