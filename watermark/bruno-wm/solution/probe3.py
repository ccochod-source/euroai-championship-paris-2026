"""Characterise the challenge watermark versus our embedding: decoder margin, band energies, JPEG blockiness,
and an amplified view of real vs our residual in flat areas."""
import sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import RELEASE, OUT
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
from probe2 import blur, SIG

def band_energy(a, region):
    x = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].cuda(); g = [blur(x, s) for s in SIG]
    return [float(((g[i] - g[i + 1])[0].permute(1, 2, 0).cpu().numpy()[region] ** 2).mean()) for i in range(len(SIG) - 1)]

def blockiness(a, region):
    g = a.astype(np.float32).mean(2); dh = np.abs(g[:, 1:] - g[:, :-1]); r = region[:, 1:] & region[:, :-1]
    cols = np.arange(1, 512)
    return [float(dh[:, cols % 8 == k][r[:, cols % 8 == k]].mean()) for k in range(8)]

for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
    marg = {"real": [], "ours": []}; E = {"real wm": [], "real clean": [], "ours wm": []}; B = {"real wm": [], "real clean": [], "ours wm": []}
    fulls = [ds[i] for i in range(len(ds)) if ds.meta[i]["full_watermark"]][:40]; cleans = [ds[i] for i in range(len(ds)) if not ds.meta[i]["full_watermark"]][:40]
    for itf, itc in zip(fulls, cleans):
        M = itf["mask"].astype(bool); s = 2.0 * itf["message"] - 1
        marg["real"].append(float((wm.logits(to_tensor(itf["image"]))[0].cpu().numpy() * s).mean()))
        E["real wm"].append(band_energy(itf["image"], M)); B["real wm"].append(blockiness(itf["image"], M))
        Mc = ~itc["mask"].astype(bool)
        E["real clean"].append(band_energy(itc["image"], Mc)); B["real clean"].append(blockiness(itc["image"], Mc))
        b = rng.integers(0, 2, wm.nbits); w = wm.embed(itc["image"], b); comp = np.where(M[..., None], w, itc["image"])   # same hole layout
        marg["ours"].append(float((wm.logits(to_tensor(comp))[0].cpu().numpy() * (2.0 * b - 1)).mean()))
        E["ours wm"].append(band_energy(w, Mc)); B["ours wm"].append(blockiness(w, Mc))
    print(f"== {meth}: decoder margin (mean logit x sign): real {np.mean(marg['real']):.3f}  ours-with-same-holes {np.mean(marg['ours']):.3f}")
    for k in E: print(f"   band energy {k:11s} " + " ".join("%8.3f" % v for v in np.median(np.array(E[k]), 0)))
    for k in B: print(f"   col-diff by x%8 {k:11s} " + " ".join("%.3f" % v for v in np.mean(np.array(B[k]), 0)))
    # amplified view: rectangles lying in flat areas
    tiles = []
    for itc in cleans:
        M = itc["mask"].astype(bool); ys, xs = np.nonzero(M)
        if len(ys) < 3000: continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if M[y0:y1, x0:x1].mean() < 0.97 or (y1 - y0) < 48 or (x1 - x0) < 48: continue
        g = itc["image"].astype(np.float32); flat = np.abs(g[y0:y1, x0 + 1:x1] - g[y0:y1, x0:x1 - 1]).mean()
        tiles.append((flat, itc, (y0, y1, x0, x1)))
    tiles.sort(key=lambda t: t[0]); rows = []
    for flat, itc, (y0, y1, x0, x1) in tiles[:4]:
        img = itc["image"]; cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        a0, b0 = int(np.clip(cy - 80, 0, 352)), int(np.clip(cx - 80, 0, 352))
        def amp(a):
            x = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None].cuda()
            return np.clip(((x - blur(x, 8.0))[0].permute(1, 2, 0).cpu().numpy()) * 12 + 128, 0, 255).astype(np.uint8)
        real = amp(img)[a0:a0 + 160, b0:b0 + 160]
        # ours: watermark the same-size rectangle somewhere in the clean part, show it the same way
        h, w = y1 - y0, x1 - x0; best = None
        for _ in range(200):
            yy, xx = int(rng.integers(0, 512 - h + 1)), int(rng.integers(0, 512 - w + 1))
            if not itc["mask"][yy:yy + h, xx:xx + w].any():
                f2 = np.abs(img[yy:yy + h, xx + 1:xx + w].astype(np.float32) - img[yy:yy + h, xx:xx + w - 1]).mean()
                if best is None or f2 < best[0]: best = (f2, yy, xx)
        if best is None: continue
        _, yy, xx = best; o = img.copy(); o[yy:yy + h, xx:xx + w] = wm.embed(np.ascontiguousarray(img[yy:yy + h, xx:xx + w]), rng.integers(0, 2, wm.nbits))
        c2, d2 = int(np.clip(yy + h // 2 - 80, 0, 352)), int(np.clip(xx + w // 2 - 80, 0, 352))
        ours = amp(o)[c2:c2 + 160, d2:d2 + 160]
        rows.append(np.concatenate([img[a0:a0 + 160, b0:b0 + 160], real, o[c2:c2 + 160, d2:d2 + 160], ours], 1))
    if rows:
        Image.fromarray(np.concatenate(rows, 0)).resize((4 * 320, len(rows) * 320), Image.Resampling.NEAREST).save(f"{OUT}/probe3_{meth}.png")
        print("   saved view with", len(rows), "rows: [real crop | real amplified | our crop | ours amplified]")
