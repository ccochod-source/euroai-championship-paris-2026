"""Probe 3: local-GRN convolutional mode vs crop decoding at the ground-truth rectangle; float16 check and speed."""
import json, time
import numpy as np, torch, torch.nn.functional as F
from common import *
import pslib

wm = pslib.patch(load_wm("pixelseal"))
torch.backends.cudnn.benchmark = True
x = torch.rand(256, 3, 256, 256, device="cuda")
for tf in [False, True]:
    torch.backends.cuda.matmul.allow_tf32 = tf; torch.backends.cudnn.allow_tf32 = tf
    pslib.crop_logits(wm, x[:8], False); torch.cuda.synchronize(); t = time.time(); a = pslib.crop_logits(wm, x, False); torch.cuda.synchronize()
    print("tf32", tf, "crops/s", 256 / (time.time() - t), "nan", torch.isnan(a).any().item())
rows = json.load(open(f"{OUT}/rect_ps_probe.json"))
items = {it["name"]: it for it in load_items("train", "pixelseal")}
d16 = []
for r in rows:
    if r["acc_bilinear"] < 0.6:
        continue
    it = items[r["name"]]; img = load_img(it["path"]); sm = sm_of(it["bits"])
    m = gt512(it["name"]); ys, xs = np.nonzero(m); y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    c = F.interpolate(img[:, :, y0:y1, x0:x1], size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
    l16 = pslib.crop_logits(wm, c, False); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; l32 = pslib.crop_logits(wm, c, False); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    d16.append(((l32 > 0) != (l16 > 0)).float().mean().item())
    line = f"{r['name']} h{h} w{w} crop {r['acc_bilinear']:.2f} |"
    for f in [1.0, 1.25, 1.5]:
        Hr = max(256, int(round(512 * 256 / h * f / 32)) * 32); Wr = max(256, int(round(512 * 256 / w * f / 32)) * 32)
        if Hr * Wr > 4200 * 4200:
            line += " (big)"; continue
        xr = F.interpolate(img, size=(Hr, Wr), mode="bilinear", align_corners=False, antialias=True)
        line += f" f{f}:"
        for k in [None, (5, 5), (9, 9)]:
            L = pslib.ppl(wm, xr, k, amp=False)[0]
            acc = ((L > 0).float() * 2 - 1 == sm[:, None, None]).float().mean(0)
            Hf, Wf = acc.shape
            gy0, gy1, gx0, gx1 = y0 / 512 * Hf, y1 / 512 * Hf, x0 / 512 * Wf, x1 / 512 * Wf
            yy, xx = torch.meshgrid(torch.arange(Hf, device="cuda") + 0.5, torch.arange(Wf, device="cuda") + 0.5, indexing="ij")
            inside = (yy > gy0) & (yy < gy1) & (xx > gx0) & (xx < gx1)
            kh, kw = max(1, min(Hf, int(round(gy1 - gy0)))), max(1, min(Wf, int(round(gx1 - gx0))))
            Lw = F.avg_pool2d(L[None], (kh, kw), stride=1)[0]
            wacc = ((Lw > 0).float() * 2 - 1 == sm[:, None, None]).float().mean(0)
            kk = int(wacc.argmax()); i, j = divmod(kk, wacc.shape[1])
            line += f" [{'g' if k is None else k[0]}] in {acc[inside].mean().item():.2f} out {acc[~inside].mean().item() if (~inside).any() else 0:.2f} win {wacc.max().item():.2f}@({i - gy0:+.1f},{j - gx0:+.1f})"
        line += " |"
    print(line, flush=True)
print("tf32 vs fp32 bit flips at GT crops: mean", np.mean(d16), "max", np.max(d16))
