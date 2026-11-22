"""Partial images (watermark on a rectangle): does the informed detector delineate the watermarked area when the template
is computed on the rectangle found by the decoding search, grown by g pixels per side (recipe: frame = mask box + 4 px)?
Zero-shot (the network only saw full-frame templates). Compares with the plain rectangle mask.
Usage: python diag_partial.py --method trustmark --ckpt out/trustmark_i1_last.pt --rects ../rects_trustmark_train.json"""
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image, downsample_mask

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--rects", required=True)
ap.add_argument("--acc", type=float, default=0.85)
a = ap.parse_args()
R = json.load(open(a.rects))
templ = Templater(load_watermark(a.method, device="cuda"))
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
ds = LocalisationDataset(RELEASE, "train", methods=[a.method])
G = (0, 2, 4, 6, 8); rows = []
for i, m in enumerate(ds.meta):
    r = R.get(m["image_name"], {})
    if m.get("full_watermark", False) or r.get("acc", 0) < a.acc: continue
    it = ds[i]; g = it["mask128"]; y0, y1, x0, x1 = r["rect"]
    x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].cuda().float(); b = torch.from_numpy(it["message"][None].astype(np.float32)).cuda()
    pm = np.zeros((512, 512), np.float32); pm[y0:y1, x0:x1] = 1
    row = [miou_image(downsample_mask(pm), g)]
    ys, xs = np.nonzero(it["mask"]); gb = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    for src, (b0, b1, c0, c1) in (("found", (y0, y1, x0, x1)), ("gt", gb)):
        for gr in G:
            f0, f1, e0, e1 = max(0, b0 - gr), min(512, b1 + gr), max(0, c0 - gr), min(512, c1 + gr)
            with torch.no_grad():
                Tc = templ(x[:, :, f0:f1, e0:e1], b); T = torch.zeros_like(x); T[:, :, f0:f1, e0:e1] = Tc / (Tc.pow(2).mean().sqrt() + 1e-3)
                with torch.autocast("cuda", dtype=torch.float16):
                    p = torch.sigmoid(net(x, T)[0].float())[0].cpu().numpy()
            inside = np.zeros((512, 512), np.float32); inside[f0:f1, e0:e1] = 1
            pred = (p > 0.5) & (downsample_mask(inside) > 0)
            row.append(miou_image(pred, g))
    rows.append(row)
A = np.array(rows)
print(f"{a.method}: {len(A)} partial train images with a decoded rectangle (acc >= {a.acc})")
print(f"  rectangle mask as is                         : {A[:, 0].mean():.4f}")
for k, src in enumerate(("found rectangle", "GT bounding box ")):
    print(f"  detector, frame = {src} + g px, g = {G} : " + "  ".join(f"{A[:, 1 + k * len(G) + j].mean():.4f}" for j in range(len(G))))
