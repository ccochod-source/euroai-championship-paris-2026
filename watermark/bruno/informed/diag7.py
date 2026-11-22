"""Per-image detector response on REAL fully watermarked images (standard template): is it bimodal?"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
meth, ckpt = sys.argv[1], sys.argv[2]
wm = load_watermark(meth, device="cuda"); templ = Templater(wm)
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rows = []
for i, m in enumerate(ds.meta):
    if not m.get("full_watermark", False): continue
    it = ds[i]; g = downsample_mask(it["mask"].astype(np.float32))
    x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].cuda().float(); b = torch.from_numpy(it["message"][None].astype(np.float32)).cuda()
    with torch.no_grad():
        t, s = normalise(templ(x, b))
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(x, t)[0].float())[0].cpu().numpy()
        lg = wm.logits(x / 255)[0].float().cpu().numpy()
    rows.append((p[g > 0].mean(), p[g == 0].mean() if (g == 0).any() else np.nan, float((lg * (it["message"] * 2.0 - 1)).mean()), float(s.mean()), float(it["image"].std())))
R = np.array(rows)
print(meth, "n =", len(R), "| detector prob in the watermarked area, per image, sorted:")
print("  ", np.round(np.sort(R[:, 0]), 2).tolist())
print("   share > 0.8: %.2f   share < 0.2: %.2f   corr(prob, decoder margin) = %.2f   corr(prob, template rms) = %.2f   corr(prob, image std) = %.2f" % (
    np.mean(R[:, 0] > .8), np.mean(R[:, 0] < .2), np.corrcoef(R[:, 0], R[:, 2])[0, 1], np.corrcoef(R[:, 0], R[:, 3])[0, 1], np.corrcoef(R[:, 0], R[:, 4])[0, 1]))
