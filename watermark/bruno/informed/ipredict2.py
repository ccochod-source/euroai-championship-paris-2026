"""Informed detector on images whose embedding frame is known: template computed on the frame (reflect-padded canvas),
probabilities 128 x 128 saved for the fusion. Images without a reliable frame (missing, or slope < --min_slope) are skipped.
Usage: python ipredict2.py --method trustmark --ckpt out/trustmark_ft2_all.pt --frames a.json,b.json --split test --out out/trustmark_ft2_testprobs.npz
frames json: {name: {"a","b","l","r"[, "slope"]}} or {name: [a, b, l, r]}"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image


def template_on_frame(templ, x, bits, fr):
    a, b, l, r = [int(v) for v in fr]
    xp, ra, rb, rl, rr = x, a, b, l, r
    while ra or rb or rl or rr:
        H, W = xp.shape[-2:]; pa, pb, pl, pr = min(ra, H - 1), min(rb, H - 1), min(rl, W - 1), min(rr, W - 1)
        xp = F.pad(xp, (pl, pr, pa, pb), mode="reflect"); ra, rb, rl, rr = ra - pa, rb - pb, rl - pl, rr - pr
    return templ(xp, bits)[:, :, a:a + 512, l:l + 512]


def load_frames(files, min_slope, lr_pad=0):
    fr = {}
    for f in files.split(","):
        if not os.path.exists(f): continue
        for k, v in json.load(open(f)).items():
            if isinstance(v, dict):
                if v.get("slope", 1.0) >= min_slope: fr[k] = [v["a"], v["b"], v.get("l", lr_pad), v.get("r", lr_pad)]
            else: fr[k] = list(v)
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--frames", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--out", required=True); ap.add_argument("--min_slope", type=float, default=0.2)
    ap.add_argument("--tta", type=int, default=1)
    a = ap.parse_args()
    fr = load_frames(a.frames, a.min_slope)
    templ = Templater(load_watermark(a.method, device="cuda"))
    net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    ds = LocalisationDataset(RELEASE, a.split, methods=[a.method]); names, probs, sc = [], [], []
    for i, m in enumerate(ds.meta):
        if m["image_name"] not in fr: continue
        it = ds[i]; x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].cuda().float(); b = torch.from_numpy(it["message"][None].astype(np.float32)).cuda()
        with torch.no_grad():
            t = normalise(template_on_frame(templ, x, b, fr[m["image_name"]]))[0]; p = 0
            views = [(0, False), (1, False), (0, True), (2, False)][:max(1, a.tta)]            # rot90 k, horizontal flip: applied to image and template together
            for k, flip in views:
                xx, tt = (x.flip(-1), t.flip(-1)) if flip else (x, t)
                xx, tt = torch.rot90(xx, k, (-2, -1)), torch.rot90(tt, k, (-2, -1))
                with torch.autocast("cuda", dtype=torch.float16):
                    q = torch.sigmoid(net(xx, tt)[0].float())
                q = torch.rot90(q, -k, (-2, -1)); p = p + (q.flip(-1) if flip else q)
            p = (p / len(views))[0].cpu().numpy()
        names.append(m["image_name"]); probs.append(p.astype(np.float16))
        if "mask128" in it: sc.append((miou_image(p > 0.5, it["mask128"]), miou_image(np.ones((128, 128)), it["mask128"])))
    np.savez_compressed(a.out, names=np.array(names), probs=np.stack(probs), frames=np.array([fr[n] for n in names]))
    if sc:
        s = np.array(sc); print(f"{a.method} {a.split}: n={len(s)} mIoU@0.5 {s[:, 0].mean():.4f} (all-ones {s[:, 1].mean():.4f})  [not out-of-fold if the checkpoint saw these images]")
    print("DONE", a.out, len(names), "images")


if __name__ == "__main__":
    main()
