"""Per REAL fully watermarked image: detector response vs size of the clean region. Are the images that match the kit
template exactly the ones without any edit?"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); templ = Templater(wm)
    net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(f"out/{meth}_i1_last.pt", map_location="cpu", weights_only=True))
    ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rows = []
    for i, m in enumerate(ds.meta):
        if not m.get("full_watermark", False): continue
        it = ds[i]; g = downsample_mask(it["mask"].astype(np.float32))
        x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].cuda().float(); b = torch.from_numpy(it["message"][None].astype(np.float32)).cuda()
        with torch.no_grad():
            t, _ = normalise(templ(x, b))
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(net(x, t)[0].float())[0].cpu().numpy()
        rows.append((1 - it["mask"].mean(), p[g > 0].mean()))
    R = np.array(rows); o = np.argsort(R[:, 0])
    print(f"\n{meth}: (clean fraction of the image : detector prob in the watermarked area), sorted by clean fraction")
    print("  " + "  ".join(f"{a:.3f}:{b:.2f}" for a, b in R[o]))
    noedit = R[:, 0] < 1e-4
    print(f"  images with NO clean region: n={noedit.sum()}  mean prob {R[noedit, 1].mean():.3f}   |   images with an edit: n={(~noedit).sum()}  mean prob {R[~noedit, 1].mean():.3f}   share > 0.8: {np.mean(R[~noedit, 1] > .8):.2f}")
