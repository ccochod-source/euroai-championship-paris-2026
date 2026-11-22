"""Post-traitements de masque validés hors échantillon (Orehn, 22/11). Part des meilleures probabilités OOF par méthode.
Options testées par méthode, choisies par validation croisée 2 moitiés (gain exigé > 0.004) :
  cc  : ne garder que les composantes connexes de taille >= cmin (masques clairsemés) ; boucher les trous < hmin (masques denses)
  box : pour un masque clairsemé, remplacer chaque grande composante par sa boîte englobante si son taux de remplissage >= r
  flat: tout à 0 si la surface prédite < amin ; tout à 1 si > amax
Sortie : orehn/out/sub_orehn_a2.pt + rapport."""
import sys, os, json, itertools
import numpy as np, torch
from scipy import ndimage as ndi
B = "/p/scratch/training2666/dasilvalopes3"; OUTB = f"{B}/wm/out"; REL = f"{B}/wm/release"
O = "/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/orehn/out"
sys.path.insert(0, f"{B}/wm/kit")
from wmloc.utils import miou_image
BASE = {"pixelseal": (["ftP6P"], 0.35), "trustmark": (["ftT6P"], 0.75), "stegastamp": (["ftP", "ftS5P"], 0.45)}
z = np.load(f"{REL}/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
meta = {s: json.load(open(f"{REL}/{s}/metadata.json")) for s in ("train", "test")}
def load(m, rs, split):
    acc = None
    for r in rs:
        z = np.load(f"{OUTB}/{m}_{r}_{split}probs.npz"); d = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
        acc = d if acc is None else {k: acc[k] + d[k] for k in acc}
    return {k: v / len(rs) for k, v in acc.items()}
def pp(p, thr, cfg):
    cmin, hmin, r, amin, amax, dthr = cfg
    m = p > thr; a = m.mean()
    if a > 0.5:                                    # image « pleine » : seuil éventuellement différent, petits trous bouchés
        m = p > (thr + dthr)
        if hmin:
            lab, n = ndi.label(~m)
            if n:
                sz = ndi.sum(~m, lab, range(1, n + 1))
                for i, s in enumerate(sz, 1):
                    if s < hmin: m[lab == i] = True
        if amax and m.mean() > amax: m[:] = True
    else:
        if cmin:
            lab, n = ndi.label(m)
            if n:
                sz = ndi.sum(m, lab, range(1, n + 1))
                for i, s in enumerate(sz, 1):
                    if s < cmin: m[lab == i] = False
        if r:
            lab, n = ndi.label(m); out = m.copy()
            for i, sl in enumerate(ndi.find_objects(lab), 1):
                comp = lab[sl] == i
                if comp.sum() >= 30 and comp.mean() >= r: out[sl] = True
            m = out
        if amin and m.mean() < amin: m[:] = False
    return m
CFGS = [(c, h, r, amin, amax, d) for c in (0, 8, 30) for h in (0, 8, 30) for r in (0, 0.7, 0.85) for amin in (0, 0.003) for amax in (0, 0.985) for d in (0.0, -0.1, 0.1)]
rep = {}; names_out, masks_out, allsc = [], [], []
for m, (runs, thr) in BASE.items():
    tr = [x["image_name"] for x in meta["train"] if x["method"] == m]; te = [x["image_name"] for x in meta["test"] if x["method"] == m]
    full = {x["image_name"]: x["full_watermark"] for x in meta["train"] if x["method"] == m}
    P = load(m, runs, "train"); T = load(m, runs, "test")
    base = {n: miou_image(P[n] > thr, gt[n]) for n in tr}
    bf = np.mean([base[n] for n in tr if full[n]]); bn = np.mean([base[n] for n in tr if not full[n]])
    S = np.array([[miou_image(pp(P[n], thr, c), gt[n]) for n in tr] for c in CFGS])        # configs x images
    idx = np.arange(len(tr)); h = [idx[0::2], idx[1::2]]
    cv = np.mean([S[int(np.argmax(S[:, h[a]].mean(1)))][h[b]].mean() for a, b in ((0, 1), (1, 0))])
    b0 = np.mean(list(base.values())); k = int(np.argmax(S.mean(1))); use = cv > b0 + 0.004
    print(f"{m}: base {b0:.4f} (pleines {bf:.4f} / partielles {bn:.4f}) | post-traitement CV {cv:.4f}, ajusté {S[k].mean():.4f} cfg {CFGS[k]} -> {'APPLIQUÉ' if use else 'non'}", flush=True)
    full_arr = np.array([full[n] for n in tr])
    print(f"   avec cfg : pleines {S[k][full_arr].mean():.4f} / partielles {S[k][~full_arr].mean():.4f}", flush=True)
    pred = [pp(T[n], thr, CFGS[k]) if use else (T[n] > thr) for n in te]
    rep[m] = dict(base=float(b0), full=float(bf), notfull=float(bn), cv=float(cv), cfg=list(CFGS[k]), used=bool(use))
    names_out += te; masks_out += pred; allsc += list(S[k] if use else base.values())
rep["overall_oof"] = float(np.mean(allsc)); print("OOF GLOBAL attendu:", round(rep["overall_oof"], 4))
M = torch.from_numpy(np.stack(masks_out)).bool(); assert M.shape == (1020, 128, 128) and len(set(names_out)) == 1020
torch.save({"image_names": names_out, "masks": M}, f"{O}/sub_orehn_a2.pt"); json.dump(rep, open(f"{O}/sub_orehn_a2.json", "w"), indent=1); print("écrit sub_orehn_a2.pt")
