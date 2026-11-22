"""Could the challenge watermark simply be weaker than wm.embed()? Decoder margin (mean logit x sign of the true bit)
as a function of the residual strength alpha on our own embedding, compared with the margin on challenge images."""
import sys, numpy as np, torch
from common import RELEASE
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
from bands import stat
REAL = {"stegastamp": 6.08, "trustmark": 7.83, "pixelseal": 8.96}
full = np.ones((512, 512), bool)
for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
    bases = [ds[i]["image"] for i in range(len(ds)) if not ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() < 0.05][:16]
    out = {}
    for c in bases:
        msg = rng.integers(0, 2, wm.nbits); s = 2.0 * msg - 1; r = wm.embed(c, msg).astype(np.float32) - c
        for al in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0):
            x = np.clip(np.round(c + al * r), 0, 255).astype(np.uint8)
            lg = wm.logits(to_tensor(x))[0].cpu().numpy()
            T = wm.embed(x, msg).astype(np.float32) - wm.embed(x, 1 - msg).astype(np.float32)
            out.setdefault(al, []).append([float((lg * s).mean()), float(((lg > 0) == msg).mean())] + [v for v, _ in stat(x.astype(np.float32), T, full)])
    print(f"== {meth}: margin on REAL full-watermark images (with ~9% holes): {REAL[meth]}")
    for al, v in out.items():
        m = np.mean(v, 0); print(f"   alpha {al:.1f}: margin {m[0]:6.2f}  bit acc {m[1]:.3f}  slopes " + " ".join("%+.3f" % t for t in m[2:]), flush=True)
