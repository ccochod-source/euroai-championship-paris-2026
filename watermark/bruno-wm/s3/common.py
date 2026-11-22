"""Shared helpers for the s3 experiments (run on JURECA inside the kit venv; see s3/README.md).

Conventions: images are float tensors B x 3 x H x W in [0,1] on the GPU; a message is a +-1 float tensor `sm`
of length nbits; masks are numpy bool arrays, 512 x 512 (pixels) or 128 x 128 (evaluation cells, 1 cell = 4 x 4 px).
"""
import json
import os
import sys

import numpy as np
import torch

W = "/p/scratch/training2666/dasilvalopes3"
KIT = f"{W}/wm/kit"
RELEASE = f"{W}/wm/release"
OUT = os.environ.get("S3_OUT", f"{W}/wm/s3/out")
METHODS = ["pixelseal", "trustmark", "stegastamp"]
MAX_ASPECT = {"pixelseal": 16.0, "stegastamp": 16.0, "trustmark": 2.0}
sys.path.insert(0, KIT)
from wmloc.utils import downsample_mask, miou_image  # noqa: E402  (official metric helpers)
from wmloc.watermarks import load_watermark  # noqa: E402

os.makedirs(OUT, exist_ok=True)


def load_wm(method, device="cuda"):
    return load_watermark(method, models_dir=f"{KIT}/models", device=device)


def load_items(split, method=None):
    """List of dicts: name, method, bits (np 0/1), full (bool or None on test), path. Masks are loaded lazily."""
    md = json.load(open(f"{RELEASE}/{split}/metadata.json"))
    out = []
    for m in md:
        if method and m["method"] != method:
            continue
        out.append(dict(name=m["image_name"], method=m["method"], bits=np.array([int(c) for c in m["message"]], np.int64),
                        full=m.get("full_watermark"), path=f"{RELEASE}/{split}/images/{m['image_name']}.png", split=split))
    return out


_M128 = {}


def gt128(split="train"):
    """dict name -> 128 x 128 bool ground-truth mask (train only)."""
    if split not in _M128:
        z = np.load(f"{RELEASE}/{split}/masks_128.npz")
        _M128[split] = {str(n): (m > 0) for n, m in zip(z["names"], z["masks"])}
    return _M128[split]


def gt512(name, split="train"):
    from PIL import Image
    return np.asarray(Image.open(f"{RELEASE}/{split}/masks/{name}.png").convert("L")) > 127


def load_img(path, device="cuda"):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255).to(device)


def sm_of(bits, device="cuda"):
    return torch.from_numpy(bits.astype(np.float32) * 2 - 1).to(device)


@torch.no_grad()
def signed(wm, imgs, sm, bs=96):
    """Decoder evidence for the known message on a batch of images (any size; the wrapper resizes to the decoder's
    working resolution). Returns (margin, acc): margin = mean over bits of logit * sign (higher = message present),
    acc = bit accuracy. Both numpy arrays of length B."""
    ms, accs = [], []
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs]
        if wm.name == "stegastamp":
            lg = wm.logits(x.clamp(0, 1), bs=32).to(sm.device)
        elif wm.name == "trustmark":
            with torch.autocast("cuda", dtype=torch.float16):
                lg = wm.logits(x)
            lg = lg.float()
        else:  # pixelseal: float16 autocast overflows to NaN on some images, keep float32
            lg = wm.logits(x).float()
        ms.append((lg * sm[None]).mean(1)); accs.append(((lg > 0).float() * 2 - 1 == sm[None]).float().mean(1))
    return torch.cat(ms).cpu().numpy(), torch.cat(accs).cpu().numpy()


def miou(pred128, gt):
    """Official per-image score. pred128, gt: 128 x 128 arrays of 0/1."""
    return float(miou_image(np.asarray(pred128).astype(np.uint8), np.asarray(gt).astype(np.uint8)))


def kind(gt):
    """Coarse type of a ground-truth 128 x 128 mask: 'rect' (exact rectangle), 'shape' (anything else)."""
    ys, xs = np.nonzero(gt)
    if len(ys) == 0:
        return "empty"
    return "rect" if gt[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() > 0.99 else "shape"
