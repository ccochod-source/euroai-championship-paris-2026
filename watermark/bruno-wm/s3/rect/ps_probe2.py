"""Probe 2: is PixelSeal's convolutional per-position logit map usable to localise the rectangle?
For strongly decodable ground-truth rectangles: run the detector convolutionally on the whole image resized so that
the rectangle maps to 256 x 256 (times a mismatch factor), compare in-rectangle vs outside agreement, and the
8 x 8 window score vs the exact crop score."""
import json, time
import numpy as np, torch, torch.nn.functional as F
from common import *

wm = load_wm("pixelseal"); det = wm.det
lin = det.pixel_decoder.linear


@torch.no_grad()
def ppl(x):
    """x: B x 3 x H x W in [0,1] (H, W multiples of 32) -> B x 256 x H/32 x W/32 per-position logits."""
    lat = det.convnext(det.preprocess(x)); emb = det.pixel_decoder.output_upscaling(lat)
    return (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:].float()


rows = json.load(open(f"{OUT}/rect_ps_probe.json"))
G = gt128(); items = {it["name"]: it for it in load_items("train", "pixelseal")}
# sanity: ppl on a 256 crop, averaged, equals det()
for r in rows:
    if r["acc_bilinear"] < 0.75:
        continue
    it = items[r["name"]]; img = load_img(it["path"]); sm = sm_of(it["bits"])
    m = gt512(it["name"]); ys, xs = np.nonzero(m); y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    line = f"{r['name']} h{h} w{w} crop_acc {r['acc_bilinear']:.2f} mar {r['mar_bilinear']:.2f} |"
    for fy, fx in [(1, 1), (0.8, 0.8), (1.25, 1.25), (1.5, 1.5), (0.6, 0.6), (1, 1.4), (1.4, 1)]:
        Hr = max(256, int(round(512 * 256 / h * fy / 32)) * 32); Wr = max(256, int(round(512 * 256 / w * fx / 32)) * 32)
        if Hr * Wr > 3500 * 3500:
            line += " (big)"; continue
        x = F.interpolate(img, size=(Hr, Wr), mode="bilinear", align_corners=False, antialias=True)
        L = ppl(x)[0]                                   # 256 x Hf x Wf
        a = (L * sm[:, None, None]).mean(0)             # per-position margin
        acc = ((L > 0).float() * 2 - 1 == sm[:, None, None]).float().mean(0)
        Hf, Wf = a.shape
        gy0, gy1, gx0, gx1 = y0 / 512 * Hf, y1 / 512 * Hf, x0 / 512 * Wf, x1 / 512 * Wf
        yy, xx = torch.meshgrid(torch.arange(Hf, device="cuda") + 0.5, torch.arange(Wf, device="cuda") + 0.5, indexing="ij")
        inside = (yy > gy0) & (yy < gy1) & (xx > gx0) & (xx < gx1)
        # window of the rectangle's size in feature cells
        kh, kw = max(1, min(Hf, int(round(gy1 - gy0)))), max(1, min(Wf, int(round(gx1 - gx0))))
        Lw = F.avg_pool2d(L[None], (kh, kw), stride=1)[0]
        wacc = ((Lw > 0).float() * 2 - 1 == sm[:, None, None]).float().mean(0)
        k = int(wacc.argmax()); i, j = divmod(k, wacc.shape[1])
        ei, ej = i - gy0, j - gx0                       # offset of the best window from the true position (cells)
        ins = acc[inside].mean().item() if inside.any() else float("nan"); outs = acc[~inside]
        line += f" f{fy}x{fx}: in {ins:.2f} out {outs.mean().item():.2f}/{outs.max().item() if outs.numel() else 0:.2f} win {wacc.max().item():.2f}@({ei:+.1f},{ej:+.1f}) |"
    print(line, flush=True)
