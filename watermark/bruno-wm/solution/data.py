"""On-the-fly synthetic composites: clean image + watermarked version blended through a random binary mask."""
import pickle, numpy as np, cv2, torch
from torch.utils.data import Dataset
from common import SYN, RELEASE
from wmloc.utils import downsample_mask

cv2.setNumThreads(0)


def blob(rng, size):
    """Smooth random blob in a size x size box (bool)."""
    k = int(rng.integers(3, 7))
    n = cv2.resize(rng.standard_normal((k, k)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1) - 0.5
    n = n - 6.0 * (yy ** 2 + xx ** 2)          # fade towards the border so the blob stays inside the box
    return n > np.quantile(n, rng.uniform(0.45, 0.8))


def shape(rng, bank, h, w):
    """Random shape resized to h x w (bool)."""
    t = rng.random()
    if t < 0.55:
        m = bank[int(rng.integers(len(bank)))].astype(np.uint8)
        if rng.random() < 0.5: m = m[:, ::-1]
        m = cv2.resize(np.ascontiguousarray(m), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    elif t < 0.85:
        m = cv2.resize(blob(rng, 64).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
    elif t < 0.93:
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (w // 2, h // 2), (max(w // 2 - 1, 1), max(h // 2 - 1, 1)), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        m = m > 0
    else:
        m = np.ones((h, w), bool)
    return m


def paste(canvas, m, cy, cx):
    """OR the shape m into canvas, centred at (cy, cx), clipped to the canvas."""
    h, w = m.shape; H, W = canvas.shape
    y0, x0 = cy - h // 2, cx - w // 2
    ya, xa = max(y0, 0), max(x0, 0); yb, xb = min(y0 + h, H), min(x0 + w, W)
    if yb > ya and xb > xa:
        canvas[ya:yb, xa:xb] |= m[ya - y0:yb - y0, xa - x0:xb - x0]


def holes(rng, bank, n, box=None, smin=20, smax=300):
    """Union of n random shapes. If box=(y0,y1,x0,x1) is given, shapes are centred inside it."""
    c = np.zeros((512, 512), bool)
    for _ in range(n):
        s = np.exp(rng.uniform(np.log(smin), np.log(smax))); a = np.exp(rng.uniform(-0.7, 0.7))
        h, w = int(np.clip(s * a, 6, 512)), int(np.clip(s / a, 6, 512))
        if box is None: cy, cx = int(rng.integers(0, 512)), int(rng.integers(0, 512))
        else: cy, cx = int(rng.integers(box[0], box[1] + 1)), int(rng.integers(box[2], box[3] + 1))
        paste(c, shape(rng, bank, h, w), cy, cx)
    return c


def make_mask(rng, bank, rect):
    """Returns (mask 512x512 bool: True = watermarked, use_rect: which watermarked version to use)."""
    y0, y1, x0, x1 = [int(v) for v in rect]
    r = rng.random()
    if r < 0.48 or y1 <= y0:                       # whole image watermarked, clean holes
        for _ in range(8):                         # real holes are small (median 6 % of the image, a quarter under 2 %)
            c = holes(rng, bank, int(rng.choice([1, 1, 1, 2, 2, 3, 4])), smin=14, smax=240)
            if rng.random() < 0.15:                # clean strips / corner boxes (outpainting-like)
                for _ in range(int(rng.integers(1, 3))):
                    t = int(rng.integers(8, 180)); side = int(rng.integers(4))
                    if side == 0: c[:t] = True
                    elif side == 1: c[-t:] = True
                    elif side == 2: c[:, :t] = True
                    else: c[:, -t:] = True
                if rng.random() < 0.4:
                    hh, ww = int(rng.integers(30, 260)), int(rng.integers(30, 260))
                    yy = 0 if rng.random() < 0.5 else 512 - hh; xx = 0 if rng.random() < 0.5 else 512 - ww
                    c[yy:yy + hh, xx:xx + ww] = True
            f = c.mean()
            if 0.001 < f < 0.5: break
        return ~c, False
    R = np.zeros((512, 512), bool); R[y0:y1, x0:x1] = True
    rh, rw = y1 - y0, x1 - x0
    if r < 0.74:                                   # plain rectangle
        return R, True
    if r < 0.985:                                  # small irregular mask on the whole-image frame (full-type image)
        for _ in range(8):
            t = rng.random()
            if t < 0.5:                            # object silhouette(s)
                m = holes(rng, bank, int(rng.integers(1, 3)), smin=24, smax=260)
            elif t < 0.75:                         # rectangle cut by a shape
                m = R & holes(rng, bank, 1, box=(y0, y1 - 1, x0, x1 - 1), smin=max(8, min(rh, rw) * 0.5), smax=max(16, max(rh, rw)))
            else:                                  # rectangle with holes
                m = R & ~holes(rng, bank, int(rng.integers(1, 3)), box=(y0, y1 - 1, x0, x1 - 1), smin=max(8, min(rh, rw) * 0.2), smax=max(16, min(rh, rw) * 1.2))
            if 64 <= m.sum() < 0.6 * 512 * 512: return m, False
        return R, True
    if r < 0.935:                                  # rectangle minus holes
        s = max(min(rh, rw), 12)
        for _ in range(8):
            m = R & ~holes(rng, bank, int(rng.integers(1, 4)), box=(y0, y1 - 1, x0, x1 - 1), smin=max(8, s * 0.2), smax=max(16, s * 1.2))
            if m.sum() > 0.25 * R.sum(): return m, True
        return R, True
    for _ in range(8):                             # object shape cut out of the rectangle (transplanted object)
        sh = shape(rng, bank, max(rh, 6), max(rw, 6))
        if rng.random() < 0.5:                     # shape fills the rectangle
            m = np.zeros((512, 512), bool); m[y0:y1, x0:x1] = sh[:rh, :rw]
        else:                                      # smaller shape somewhere inside
            m = R & holes(rng, bank, 1, box=(y0, y1 - 1, x0, x1 - 1), smin=max(8, min(rh, rw) * 0.5), smax=max(16, max(rh, rw)))
        if m.sum() >= 64: return m, True
    return R, True


class Synth(Dataset):
    def __init__(self, method, tags=("a",), n=None, seed=0, p_in=0.0, pairs=False):
        self.p_in = p_in; self.pairs = pairs
        import os
        new = any(t in tags for t in "bcdef")
        ind = "in2" if new else "in"                   # tags "b"/"c" = frame model (gen_wm2.py / gen_wm3.py)
        if p_in > 0:                                   # in-domain bases (see gen_indomain.py / gen_wm2.py --indomain)
            self.ib = np.load(f"{SYN}/{method}_{ind}_base.npy", mmap_mode="r")
            IN = {"c": "in3", "d": "in4", "e": "in5", "f": "in6"}
            self.ifulls = [np.load(f"{SYN}/{method}_{IN.get(t, ind)}_full.npy", mmap_mode="r") for t in tags]
            self.ifull = self.ifulls[0]
            self.irect = np.load(f"{SYN}/{method}_{ind}_rect.npy", mmap_mode="r")
            self.irects = np.load(f"{SYN}/{method}_{ind}_rects.npy")
        self.clean = np.load(f"{SYN}/clean_b.npy" if new else f"{SYN}/clean.npy", mmap_mode="r")
        rt = lambda t: "b" if t in ("c", "d", "e", "f") else t   # rect-type data of the canvas tags is the one of tag "b"
        self.full = [np.load(f"{SYN}/{method}_full_{t}.npy", mmap_mode="r") for t in tags]
        self.rect = [np.load(f"{SYN}/{method}_rect_{rt(t)}.npy", mmap_mode="r") for t in tags]
        self.rects = [np.load(f"{SYN}/{method}_rects_{rt(t)}.npy") for t in tags]
        self.bank = pickle.load(open(f"{SYN}/shape_bank.pkl", "rb"))
        self.nc = min([len(self.clean)] + [len(f) for f in self.full])    # full-type arrays may cover only the first images
        self.n = n or len(self.clean); self.seed = seed; self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, self.epoch, i, info.id if info else 0, int(torch.randint(0, 2**31 - 1, (1,)))])
        if self.p_in > 0 and rng.random() < self.p_in:
            j = int(rng.integers(len(self.ib))); k = int(rng.integers(self.irect.shape[1]))
            m, use_rect = make_mask(rng, self.bank, self.irects[j, k])
            c = np.array(self.ib[j])
            fl = self.ifulls[int(rng.integers(len(self.ifulls)))]
            w = np.array(self.irect[j, k] if use_rect else fl[j, int(rng.integers(fl.shape[1]))])
        else:
            j = int(rng.integers(self.nc)); t = int(rng.integers(len(self.full)))
            m, use_rect = make_mask(rng, self.bank, self.rects[t][j])
            c = np.array(self.clean[j]); w = np.array((self.rect if use_rect else self.full)[t][j])
        y = torch.from_numpy(downsample_mask(m.astype(np.float32))).float()
        if self.pairs:
            return torch.from_numpy(c).permute(2, 0, 1), torch.from_numpy(w).permute(2, 0, 1), torch.from_numpy(m), y
        img = np.where(m[..., None], w, c)
        return torch.from_numpy(img).permute(2, 0, 1), y


class Real(Dataset):
    """Challenge images (train split has masks)."""
    def __init__(self, split, method):
        from wmloc.data import LocalisationDataset
        self.ds = LocalisationDataset(RELEASE, split, methods=[method])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        it = self.ds[i]
        m = torch.from_numpy(it["mask128"]).float() if "mask128" in it else torch.zeros(128, 128)
        return torch.from_numpy(it["image"]).permute(2, 0, 1), m, it["image_name"], bool(it.get("full_watermark", False))
