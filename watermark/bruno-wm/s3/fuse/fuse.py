"""Fusion / decision layer. Builds one 128 x 128 mask per image from the available components and scores the policy
on the real training images (official metric), by method and image type.

Components (each optional):
  whole   out/whole_<split>.npz                      whole-image bit accuracy (mine)
  unet    <W>/wm/out/<method>_<tag>_<split>probs.npz probability maps of session 65 (train = out-of-fold)
  rect    <T>/bruno/rects_<method>_<split>.json      rectangles of session 75 (rect, acc)
  holes   out/<prefix>_<method>_<split>.npz          evidence maps for whole-frame images (higher = watermarked)
usage: fuse.py [--holes PREFIX] [--hole_tau 0.3] [--write NAME]
"""
import argparse, json, os, numpy as np, torch
from common import *
T = "/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task"
ap = argparse.ArgumentParser()
ap.add_argument("--holes", default=""); ap.add_argument("--hole_tau", default="0.3"); ap.add_argument("--write", default="")
ap.add_argument("--rect_first", default="", help="methods where a verified rectangle overrides the network map")
ap.add_argument("--unet_partial", default="", help="methods whose network map is used only on images that are not whole-frame")
ap.add_argument("--ens", default="", help="method:npz_in_OUT_with_{split}:weight:min_slope[:thr] - average a second map into the network map on whole-frame images")
ap.add_argument("--shrink", type=int, default=0, help="shrink found rectangles by this many px per side (frame = mask + 4 px)")
ap.add_argument("--rects", default="", help="extra rectangle dirs, e.g. pixelseal:/path/rects_pixelseal_{split}.json:0.62")
ap.add_argument("--unet", default="stegastamp:ftA:0.5"); ap.add_argument("--full_thr", type=float, default=0.70); ap.add_argument("--rect_acc", type=float, default=0.85)
a = ap.parse_args()
UNET = {s.split(":")[0]: (s.split(":")[1], float(s.split(":")[2])) for s in a.unet.split(",") if s}
TAU = {m: float(a.hole_tau) for m in METHODS} if ":" not in a.hole_tau else {s.split(":")[0]: float(s.split(":")[1]) for s in a.hole_tau.split(",")}
ENS = {s.split(":")[0]: s.split(":")[1:] for s in a.ens.split(",") if s}
RECTS = {s.split(":")[0]: (s.split(":")[1], float(s.split(":")[2])) for s in a.rects.split(",") if s}


def npz_maps(path):
    if not os.path.exists(path): return {}
    z = np.load(path); key = "probs" if "probs" in z else "maps"
    return {str(n): m for n, m in zip(z["names"], z[key])}


def predict(split):
    wh = np.load(f"{OUT}/whole_{split}.npz"); acc = {str(n): float(v) for n, v in zip(wh["names"], wh["acc"])}
    pred, src = {}, {}
    for meth in METHODS:
        items = load_items(split, meth)
        un = {}
        if meth in UNET:
            pth = lambda t: f"{OUT}/{t[1:]}_{meth}_{split}.npz" if t.startswith("@") else f"{W}/wm/out/{meth}_{t}_{split}probs.npz"
            parts = [npz_maps(pth(t)) for t in UNET[meth][0].split("+")]
            if parts[0]: un = {n: np.mean([q[n].astype(np.float32) for q in parts if n in q], 0) for n in parts[0]}   # first map is the reference; others join where they exist
        ho = npz_maps(f"{OUT}/{a.holes}_{meth}_{split}.npz") if a.holes else {}
        rj = f"{T}/bruno/rects_{meth}_{split}.json"; racc = a.rect_acc
        if meth in RECTS: rj = RECTS[meth][0].format(split=split); racc = RECTS[meth][1]
        rc = json.load(open(rj)) if os.path.exists(rj) else {}
        en = {}
        if meth in ENS:
            ef = f"{OUT}/" + ENS[meth][0].format(split=split)
            if os.path.exists(ef):
                z = np.load(ef); en = {str(k): (m, float(sl)) for k, m, sl in zip(z["names"], z["probs"], z["slope"])}
        for it in items:
            n = it["name"]
            if meth in a.rect_first.split(",") and acc[n] <= 0.65 and n in rc and "rect" in rc[n] and rc[n].get("acc", 0) >= racc:
                y0, y1, x0, x1 = [int(round(float(v))) for v in rc[n]["rect"]]; m = np.zeros((512, 512), np.float32); m[y0:y1, x0:x1] = 1
                pred[n] = np.asarray(downsample_mask(m)) > 0; src[n] = "rect"
            elif n in un and not (meth in a.unet_partial.split(",") and (acc[n] > a.full_thr or (n in rc and "rect" in rc[n] and rc[n].get("acc", 0) >= racc))):
                p = un[n].astype(np.float32); thr = UNET[meth][1]; src[n] = "unet"
                if n in en and acc[n] > a.full_thr and en[n][1] > float(ENS[meth][2]):
                    w = float(ENS[meth][1]); p = (1 - w) * p + w * en[n][0].astype(np.float32); src[n] = "unet+informed"
                    if len(ENS[meth]) > 3: thr = float(ENS[meth][3])
                pred[n] = p > thr
            elif acc[n] > a.full_thr:
                if n in ho:
                    e = ho[n].astype(np.float32); pred[n] = e > TAU.get(meth, 0.3) * np.median(e); src[n] = "holes"
                else:
                    pred[n] = np.ones((128, 128), bool); src[n] = "ones"
            elif n in rc and "rect" in rc[n] and rc[n].get("acc", 0) >= racc:
                y0, y1, x0, x1 = [int(round(float(v))) for v in rc[n]["rect"]]; k = a.shrink if (y1 - y0 > 2 * a.shrink + 4 and x1 - x0 > 2 * a.shrink + 4) else 0
                m = np.zeros((512, 512), np.float32); m[max(0, y0 + k):y1 - k, max(0, x0 + k):x1 - k] = 1
                pred[n] = np.asarray(downsample_mask(m)) > 0; src[n] = "rect"
            else:
                pred[n] = np.zeros((128, 128), bool); src[n] = "zeros"
    return pred, src


pred, src = predict("train"); g = gt128(); tot = []
print(f"policy: unet={UNET} holes={a.holes or '-'} tau={TAU} full_thr={a.full_thr} rect_acc={a.rect_acc} rects={RECTS}")
for meth in METHODS:
    rows = [(("full" if it["full"] else kind(g[it["name"]])), miou(pred[it["name"]], g[it["name"]]), src[it["name"]]) for it in load_items("train", meth)]
    sc = np.array([r[1] for r in rows]); ty = np.array([r[0] for r in rows]); tot.append(sc.mean())
    print(f"  {meth:10s} {sc.mean():.4f} | " + "  ".join(f"{t} {sc[ty == t].mean():.3f} (n={np.sum(ty == t)})" for t in ["full", "rect", "shape"])
          + " | sources: " + ", ".join(f"{s}={sum(r[2] == s for r in rows)}" for s in sorted(set(r[2] for r in rows))))
print(f"  TRAIN OVERALL {np.mean(tot):.4f}")
if a.write:
    pt, st = predict("test"); names = [it["name"] for it in load_items("test")]
    masks = torch.from_numpy(np.stack([pt[n] for n in names])).bool()
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    out = f"{OUT}/sub_{a.write}.pt"; torch.save({"image_names": names, "masks": masks}, out)
    print("WROTE", out, "sources:", {s: sum(v == s for v in st.values()) for s in sorted(set(st.values()))}, f"mean mask {masks.float().mean():.3f}")
