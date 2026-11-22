"""Exp 1: compute candidate localisation maps on train (or test) and save them.
Maps (all returned at 128x128, higher = more likely watermarked):
  win{s}   : sliding-window decode, signed bit agreement with the known message, averaged over covering windows
  pp       : per-position logits (pixelseal/trustmark) dotted with the signed message
  grad     : |d BCE(logits, msg) / d image| smoothed (pixelseal/trustmark)
Usage: python exp1_maps.py --kit <starter dir> --data <release dir> --split train --out maps_train.npz
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--methods", default="pixelseal,trustmark,stegastamp")
a = ap.parse_args()
sys.path.insert(0, a.kit)
from wmloc.data import LocalisationDataset  # noqa
from wmloc.watermarks import load_watermark  # noqa

dev = "cuda"
WINDOWS = [(128, 64), (192, 64), (256, 128)]


def windows(size, stride, H=512):
    pos = list(range(0, H - size + 1, stride))
    if pos[-1] != H - size:
        pos.append(H - size)
    return [(y, x) for y in pos for x in pos]


def sliding_maps(wm, img, sm):
    """img: 1x3x512x512 float on dev; sm: nbits signed message (+1/-1) on cpu tensor."""
    out = {}
    for s, st in WINDOWS:
        coords = windows(s, st)
        crops = torch.cat([img[:, :, y:y + s, x:x + s] for y, x in coords])
        lg = wm.logits(crops).float().cpu()                      # n x nbits
        agree = (torch.tanh(lg) * sm[None]).mean(1)              # soft agreement in [-1, 1]
        acc = torch.zeros(512, 512); cnt = torch.zeros(512, 512)
        for (y, x), v in zip(coords, agree):
            acc[y:y + s, x:x + s] += v; cnt[y:y + s, x:x + s] += 1
        out[f"win{s}"] = F.avg_pool2d((acc / cnt)[None, None], 4)[0, 0].numpy()
    return out


def pp_and_grad(wm, img, sm):
    out = {}
    f = wm.features(img)
    pp = f["per_position_logits"][0].float()                     # nbits x h x w
    m = (pp * sm.to(pp.device)[:, None, None]).mean(0)[None, None]
    out["pp"] = F.interpolate(m, size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    x = img.clone().requires_grad_(True)
    with torch.enable_grad():
        if wm.name == "pixelseal":
            lg = wm.det(wm._resize(x))[:, 1:]
        else:
            lg = wm.net(wm._prep(x))
        tgt = (sm.to(lg.device) > 0).float()[None]
        loss = F.binary_cross_entropy_with_logits(lg.float(), tgt)
        g, = torch.autograd.grad(loss, x)
    gm = g.abs().sum(1, keepdim=True)
    gm = F.avg_pool2d(gm, 4)                                      # 128x128
    gm = F.avg_pool2d(gm, 9, stride=1, padding=4)
    out["grad"] = gm[0, 0].cpu().numpy()
    return out


ds = LocalisationDataset(a.data, a.split)
methods = a.methods.split(",")
res = {"names": [], "method": [], "gt": [], "full_acc": []}
maps = {}
for meth in methods:
    wm = load_watermark(meth, models_dir=Path(a.kit) / "models", device=dev)
    idx = [i for i, m in enumerate(ds.meta) if m["method"] == meth]
    if a.limit:
        idx = idx[:a.limit]
    t0 = time.time()
    for k, i in enumerate(idx):
        it = ds[i]
        img = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(dev)
        sm = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1)
        mp = sliding_maps(wm, img, sm)
        full = (torch.tanh(wm.logits(img).float().cpu()) * sm[None]).mean().item()
        if meth != "stegastamp":
            mp.update(pp_and_grad(wm, img, sm))
        for kk, v in mp.items():
            maps.setdefault(kk, {})[it["image_name"]] = v.astype(np.float32)
        res["names"].append(it["image_name"]); res["method"].append(meth); res["full_acc"].append(full)
        res["gt"].append(it.get("mask128", np.zeros((128, 128), np.uint8)))
        if k % 20 == 0:
            print(meth, k, len(idx), f"{time.time() - t0:.0f}s", flush=True)
    del wm; torch.cuda.empty_cache()

save = {"names": np.array(res["names"]), "method": np.array(res["method"]),
        "gt": np.stack(res["gt"]), "full_acc": np.array(res["full_acc"])}
for kk, d in maps.items():
    save["map_" + kk] = np.stack([d.get(n, np.full((128, 128), np.nan, np.float32)) for n in res["names"]])
np.savez_compressed(a.out, **save)
print("saved", a.out, list(save))
