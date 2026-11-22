"""Fusion masque du réseau + rectangle trouvé par la recherche de Bruno (lecture seule de ses fichiers), validée OOF.
Pour une image avec un rectangle confiant (acc >= a) : 'rep' = remplacer par le rectangle, 'and' = réseau ET rectangle,
'or' = réseau OU rectangle, 'repfill' = rectangle si le réseau le remplit à >= 60 %, sinon réseau ET rectangle.
Choix par validation croisée 2 moitiés (gain exigé > 0.004). Sortie : orehn/out/sub_orehn_a3.pt."""
import sys, os, json
import numpy as np, torch
B = "/p/scratch/training2666/dasilvalopes3"; OUTB = f"{B}/wm/out"; REL = f"{B}/wm/release"
BR = "/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/bruno"; S3 = f"{B}/wm/s3/out"
O = "/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/orehn/out"
sys.path.insert(0, f"{B}/wm/kit")
from wmloc.utils import miou_image
BASE = {"pixelseal": (["ftP6P"], 0.35), "trustmark": (["ftT6P"], 0.75), "stegastamp": (["ftP", "ftS5P"], 0.45)}
RECT = {"trustmark": (f"{BR}/rects_trustmark_train.json", f"{BR}/rects_trustmark_test.json"),
        "pixelseal": (f"{S3}/rects_pixelseal_train.json", f"{S3}/rects_pixelseal_test.json"),
        "stegastamp": (f"{S3}/rects_stegastamp_train.json", f"{S3}/rects_stegastamp_test.json")}
z = np.load(f"{REL}/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
meta = {s: json.load(open(f"{REL}/{s}/metadata.json")) for s in ("train", "test")}
def load(m, rs, split):
    acc = None
    for r in rs:
        z = np.load(f"{OUTB}/{m}_{r}_{split}probs.npz"); d = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
        acc = d if acc is None else {k: acc[k] + d[k] for k in acc}
    return {k: v / len(rs) for k, v in acc.items()}
def rmask(rect):
    y0, y1, x0, x1 = rect; m = np.zeros((128, 128), bool)
    m[max(0, int(round(y0 / 4))):min(128, int(round(y1 / 4))), max(0, int(round(x0 / 4))):min(128, int(round(x1 / 4)))] = True; return m
def fuse(net, info, mode, a, key):
    if not info or key not in info or info.get("acc", 0) < a or net.mean() > 0.5: return net
    r = rmask(info[key])
    if r.sum() == 0: return net
    if mode == "rep": return r
    if mode == "and": return net & r
    if mode == "or": return net | r
    if mode == "repfill": return r if (net & r).sum() >= 0.6 * r.sum() else (net & r)
    return net
rep = {}; names_out, masks_out, allsc = [], [], []
for m, (runs, thr) in BASE.items():
    tr = [x["image_name"] for x in meta["train"] if x["method"] == m]; te = [x["image_name"] for x in meta["test"] if x["method"] == m]
    P = load(m, runs, "train"); T = load(m, runs, "test")
    Rtr = json.load(open(RECT[m][0])) if os.path.exists(RECT[m][0]) else {}; Rte = json.load(open(RECT[m][1])) if os.path.exists(RECT[m][1]) else {}
    ntr = sum("rect" in v for v in Rtr.values()); nte = sum("rect" in v for v in Rte.values())
    base = np.array([miou_image(P[n] > thr, gt[n]) for n in tr]); idx = np.arange(len(tr)); h = [idx[0::2], idx[1::2]]
    cfgs = [(mode, a, key) for mode in ("rep", "and", "or", "repfill") for a in (0.8, 0.9, 0.95, 0.99) for key in ("rect", "rect_hc")]
    S = np.array([[miou_image(fuse(P[n] > thr, Rtr.get(n), *c), gt[n]) for n in tr] for c in cfgs]) if ntr else base[None]
    k = int(np.argmax(S.mean(1))); cv = np.mean([S[int(np.argmax(S[:, h[a]].mean(1)))][h[b]].mean() for a, b in ((0, 1), (1, 0))])
    use = bool(ntr and nte and cv > base.mean() + 0.004)
    hasr = np.array([("rect" in Rtr.get(n, {})) for n in tr])
    print(f"{m}: rectangles train {ntr} / test {nte} | base {base.mean():.4f} | fusion CV {cv:.4f}, ajustée {S[k].mean():.4f} cfg {cfgs[k] if ntr else None} -> {'APPLIQUÉE' if use else 'non'}"
          + (f" | sur les {hasr.sum()} images avec rectangle : base {base[hasr].mean():.4f} -> {S[k][hasr].mean():.4f}" if hasr.any() else ""), flush=True)
    pred = [fuse(T[n] > thr, Rte.get(n), *cfgs[k]) if use else (T[n] > thr) for n in te]
    rep[m] = dict(base=float(base.mean()), cv=float(cv), cfg=list(cfgs[k]) if ntr else None, used=use, n_rect_train=int(ntr), n_rect_test=int(nte))
    names_out += te; masks_out += pred; allsc += list(S[k] if use else base)
rep["overall_oof"] = float(np.mean(allsc)); print("OOF GLOBAL attendu:", round(rep["overall_oof"], 4))
M = torch.from_numpy(np.stack(masks_out)).bool(); assert M.shape == (1020, 128, 128) and len(set(names_out)) == 1020
torch.save({"image_names": names_out, "masks": M}, f"{O}/sub_orehn_a3.pt"); json.dump(rep, open(f"{O}/sub_orehn_a3.json", "w"), indent=1); print("écrit sub_orehn_a3.pt")
