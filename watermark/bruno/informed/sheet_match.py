"""Contact sheet: REAL fully watermarked TrustMark images whose residual matches the kit template (top rows) vs
images that do not (bottom rows). Each tile: image with the clean region outlined in red."""
import os, sys, numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask
meth = sys.argv[1]
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
    rows.append((float(p[g > 0].mean()), i, it["image_name"]))
rows.sort(reverse=True)
def tile(i, pr):
    it = ds[i]; a = it["image"].copy(); hole = (it["mask"] == 0).astype(np.uint8)
    cs, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(a, cs, -1, (255, 0, 0), 3)
    a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA); cv2.putText(a, f"{pr:.2f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2); return a
top = [tile(i, p) for p, i, _ in rows[:12]]; bot = [tile(i, p) for p, i, _ in rows[-24:]]
sheet = np.concatenate([np.concatenate(top[:6], 1), np.concatenate(top[6:12], 1), np.zeros((12, 1536, 3), np.uint8), np.concatenate(bot[:6], 1), np.concatenate(bot[6:12], 1), np.concatenate(bot[12:18], 1), np.concatenate(bot[18:24], 1)], 0)
Image.fromarray(sheet).save(f"out/sheet_match_{meth}.jpg", quality=85)
print("matched names:", [n for p, i, n in rows[:16]]); print("saved")
