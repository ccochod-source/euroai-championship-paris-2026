"""Exp 2 (real train images): do message-aware, training-free signals localise the watermark?
A. data stats per method: full flag, is the mask a plain rectangle (== its bounding box)?, bbox size.
B. bit accuracy when decoding the whole image vs the crop to the GT mask bounding box (oracle rectangle).
C. matched filter by re-embedding the known message:
     R(X, m) = embed(X, m) - X ; template D = R(X, m) - mean_k R(X, m_k) (random messages m_k) ; HP = X - blur(X)
     statistic = local normalised correlation of HP with the template.
   full images: re-embed the whole image. notfull images: re-embed the oracle bbox crop (upper bound for a rect search).
Usage: python exp2_resid.py --kit . --data release --out bruno/exp2.npz [--limit N]
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--methods", default="pixelseal,trustmark,stegastamp")
ap.add_argument("--K", type=int, default=3)
a = ap.parse_args()
sys.path.insert(0, a.kit)
from wmloc.data import LocalisationDataset  # noqa
from wmloc.watermarks import load_watermark  # noqa
from wmloc.utils import to_tensor, miou_image  # noqa


def auc(score, label):
    label = np.asarray(label).astype(bool).ravel(); score = np.asarray(score).ravel()
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bitacc(wm, img, msg):
    lg = wm.logits(to_tensor(np.ascontiguousarray(img)))[0].float().cpu().numpy()
    return float(((lg > 0).astype(int) == msg).mean())


def mf_maps(wm, crop, bits, rng, K):
    crop = np.ascontiguousarray(crop); c = crop.astype(np.float32)
    R = wm.embed(crop, bits).astype(np.float32) - c
    Rk = np.mean([wm.embed(crop, rng.integers(0, 2, wm.nbits)).astype(np.float32) - c for _ in range(K)], 0)
    D = R - Rk
    out = {}
    for hn, sig in (("g15", 1.5), ("g08", 0.8)):
        HP = c - cv2.GaussianBlur(c, (0, 0), sig)
        eh = (HP * HP).sum(2)
        for tn, T in (("D", D), ("R", R)):
            s = (HP * T).sum(2); et = (T * T).sum(2)
            for k in (17, 33):
                out[f"{tn}_{hn}_k{k}"] = cv2.blur(s, (k, k)) / (np.sqrt(cv2.blur(et, (k, k)) * cv2.blur(eh, (k, k))) + 1e-6)
    return out, float(np.abs(R).mean()), float(np.abs(D).mean())


def to128(m):
    return m.reshape(128, 4, 128, 4).mean((1, 3))


rng = np.random.default_rng(0)
rows, maps = [], {}
for meth in a.methods.split(","):
    wm = load_watermark(meth, models_dir=Path(a.kit) / "models", device="cuda")
    ds = LocalisationDataset(a.data, "train", methods=[meth])
    n = min(len(ds), a.limit) if a.limit else len(ds)
    if meth != "stegastamp":
        f = wm.features(to_tensor(ds[0]["image"]))
        print(meth, "feature shapes:", {k: tuple(v.shape) for k, v in f.items()}, flush=True)
    t0 = time.time()
    for i in range(n):
        it = ds[i]; img, mask, msg = it["image"], it["mask"], it["message"]
        full = bool(it.get("full_watermark", False))
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            y0, y1, x0, x1 = 0, 512, 0, 512
        else:
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        fill = float(mask[y0:y1, x0:x1].mean()) if len(ys) else 0.0
        r = dict(name=it["image_name"], method=meth, full=full, frac=float(mask.mean()), bbox=[y0, y1, x0, x1], fill=fill,
                 acc_whole=bitacc(wm, img, msg))
        try:
            r["acc_bbox"] = bitacc(wm, img[y0:y1, x0:x1], msg) if (y1 - y0 >= 16 and x1 - x0 >= 16) else float("nan")
        except Exception as e:
            r["acc_bbox"] = float("nan"); r["err_dec"] = repr(e)[:100]
        try:
            if full:
                mm, r["absR"], r["absD"] = mf_maps(wm, img, msg, rng, a.K)
                full_maps = mm
            else:
                mm, r["absR"], r["absD"] = mf_maps(wm, img[y0:y1, x0:x1], msg, rng, a.K)
                full_maps = {}
                for k, v in mm.items():
                    cvs = np.full((512, 512), -1.0, np.float32); cvs[y0:y1, x0:x1] = v; full_maps[k] = cvs
            g = it["mask128"]
            inside = np.zeros((512, 512), bool); inside[y0:y1, x0:x1] = True
            ins128 = to128(inside.astype(np.float32)) > 0.5
            for k, v in full_maps.items():
                v128 = to128(v.astype(np.float32))
                maps.setdefault(k, {})[it["image_name"]] = v128.astype(np.float32)
                r["auc_" + k] = auc(v128, g) if full else auc(v128[ins128], g[ins128])
        except Exception as e:
            r["err_mf"] = repr(e)[:160]
        rows.append(r)
        if i % 25 == 0:
            print(meth, i, n, f"{time.time() - t0:.0f}s", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                                          if k in ("full", "frac", "fill", "acc_whole", "acc_bbox", "auc_D_g15_k17", "err_mf")}, flush=True)
    del wm; torch.cuda.empty_cache()

names = [r["name"] for r in rows]
save = {"names": np.array(names), "rows": np.array(json.dumps(rows))}
for k, d in maps.items():
    save["map_" + k] = np.stack([d.get(nm, np.full((128, 128), np.nan, np.float32)) for nm in names])
np.savez_compressed(a.out, **save)

# ---------------------------------------------------------------- summary
import collections
print("\n===== SUMMARY =====")
for meth in a.methods.split(","):
    R = [r for r in rows if r["method"] == meth]
    for flag in (True, False):
        S = [r for r in R if r["full"] == flag]
        if not S:
            continue
        g = lambda key: np.array([r.get(key, np.nan) for r in S], dtype=float)
        print(f"\n[{meth}] full={flag} n={len(S)} frac_wm mean={g('frac').mean():.3f}  plain-rect(fill>.98)={np.mean(g('fill') > .98):.2f} "
              f"fill mean={g('fill').mean():.2f}")
        print(f"   bit acc whole image: mean={np.nanmean(g('acc_whole')):.3f} (q10={np.nanquantile(g('acc_whole'), .1):.2f})   "
              f"bit acc on GT bbox crop: mean={np.nanmean(g('acc_bbox')):.3f} (q10={np.nanquantile(g('acc_bbox'), .1):.2f} q50={np.nanmedian(g('acc_bbox')):.2f})")
        print(f"   |R| mean={np.nanmean(g('absR')):.2f}  |D| mean={np.nanmean(g('absD')):.2f}  mf errors={sum('err_mf' in r for r in S)}")
        for k in sorted(maps):
            au = g("auc_" + k)
            print(f"   AUC {k:12s} mean={np.nanmean(au):.3f} median={np.nanmedian(au):.3f} (n valid={np.isfinite(au).sum()})")
