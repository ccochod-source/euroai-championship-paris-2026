"""Synthetic composites of FULLY watermarked images (whole image watermarked with a known message, clean holes):
returns image, 128 x 128 mask, message bits and the residual actually present (mask * (watermarked - clean))."""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from common import SYN, RELEASE
from data import make_mask
from wmloc.utils import downsample_mask


class SynthFull(Dataset):
    def __init__(self, method, tag="a", n=10 ** 9, seed=0, p_allwm=0.06):
        self.clean = np.load(f"{SYN}/clean.npy", mmap_mode="r")
        self.full = np.load(f"{SYN}/{method}_full_{tag}.npy", mmap_mode="r")
        self.msgs = np.load(f"{SYN}/{method}_msgs_{tag}.npy")
        # shape_bank.pkl is our own file (written by gen_masks.py from COCO annotations), not third-party data
        self.bank = pickle.load(open(f"{SYN}/shape_bank.pkl", "rb"))
        self.n, self.seed, self.p_allwm = n, seed, p_allwm

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, i, info.id if info else 0, int(torch.randint(0, 2 ** 31 - 1, (1,)))])
        j = int(rng.integers(len(self.clean)))
        if rng.random() < self.p_allwm:
            m = np.ones((512, 512), bool)                      # no edit at all (about 7 % of the real full images)
        else:
            m, _ = make_mask(rng, self.bank, (0, 0, 0, 0))     # rect with y1 <= y0 -> "whole image, clean holes" branch
        c = np.array(self.clean[j]); w = np.array(self.full[j])
        img = np.where(m[..., None], w, c)
        resid = np.clip((w.astype(np.int16) - c) * m[..., None], -127, 127).astype(np.int8)
        return (torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(downsample_mask(m.astype(np.float32))).float(),
                torch.from_numpy(self.msgs[j, 0].astype(np.float32)), torch.from_numpy(resid).permute(2, 0, 1))


class Real(Dataset):
    """Challenge images of one method: image, mask128 (zeros for test), bits, name, full flag (False for test)."""
    def __init__(self, split, method):
        from wmloc.data import LocalisationDataset
        self.ds = LocalisationDataset(RELEASE, split, methods=[method])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        it = self.ds[i]
        m = torch.from_numpy(it["mask128"]).float() if "mask128" in it else torch.zeros(128, 128)
        return (torch.from_numpy(it["image"]).permute(2, 0, 1), m, torch.from_numpy(it["message"].astype(np.float32)),
                it["image_name"], bool(it.get("full_watermark", False)))
