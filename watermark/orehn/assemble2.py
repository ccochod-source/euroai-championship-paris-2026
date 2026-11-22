"""Assemblage rapide (Orehn, 22/11) à partir des probabilités hors échantillon déjà calculées par Bruno (lecture seule).
Pour chaque méthode : score OOF (mIoU officiel) de chaque run, ensembles gloutons (moyenne), seuils par régime
(image presque pleine / région petite, décidé sur la probabilité moyenne), nettoyage morphologique optionnel.
Écrit la soumission et un rapport dans orehn/out/. Ne modifie rien chez Bruno."""
import sys, os, json, glob, itertools
import numpy as np, torch
from scipy import ndimage as ndi
B = "/p/scratch/training2666/dasilvalopes3"; OUTB = f"{B}/wm/out"; REL = f"{B}/wm/release"
O = "/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/orehn/out"; os.makedirs(O, exist_ok=True)
sys.path.insert(0, f"{B}/wm/kit")
from wmloc.utils import miou_image
METHODS = ["pixelseal", "trustmark", "stegastamp"]
z = np.load(f"{REL}/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
meta = {s: json.load(open(f"{REL}/{s}/metadata.json")) for s in ("train", "test")}
def load(m, r, split):
    z = np.load(f"{OUTB}/{m}_{r}_{split}probs.npz"); return dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
def score(P, names, thr):
    return float(np.mean([miou_image(P[n] > thr, gt[n]) for n in names]))
def best_thr(P, names, grid=np.round(np.arange(0.05, 0.96, 0.05), 2)):
    s = [(score(P, names, t), float(t)) for t in grid]; return max(s)
def post(mask, k):
    if k == 0: return mask
    m = ndi.binary_opening(mask, iterations=k); m = ndi.binary_closing(m, iterations=k); return m
def pred2(P, n, tl, th, cut, k):
    p = P[n]; t = th if p.mean() > cut else tl
    return post(p > t, k)
rep = {}; names_out, masks_out, allsc_base, allsc_new = [], [], [], []
for m in METHODS:
    tr = [x["image_name"] for x in meta["train"] if x["method"] == m]; te = [x["image_name"] for x in meta["test"] if x["method"] == m]
    runs = sorted({os.path.basename(p)[len(m) + 1:-len("_trainprobs.npz")] for p in glob.glob(f"{OUTB}/{m}_*_trainprobs.npz")}
                  & {os.path.basename(p)[len(m) + 1:-len("_testprobs.npz")] for p in glob.glob(f"{OUTB}/{m}_*_testprobs.npz")})
    PT = {}; single = []
    for r in runs:
        try:
            P = load(m, r, "train")
            if not all(n in P for n in tr): continue
            PT[r] = P; s, t = best_thr(P, tr); single.append((s, r, t))
        except Exception as e:
            print("skip", m, r, repr(e)[:80])
    single.sort(reverse=True); print(m, "runs seuls:", [(round(s, 4), r, t) for s, r, t in single], flush=True)
    # ensemble glouton
    cur = [single[0][1]]; cs = single[0][0]
    avg = lambda rs: {n: np.mean([PT[r][n] for r in rs], 0) for n in tr}
    improved = True
    while improved and len(cur) < 4:
        improved = False
        for r in [x[1] for x in single if x[1] not in cur]:
            s, t = best_thr(avg(cur + [r]), tr)
            if s > cs + 0.002: cs, add, improved = s, r, True
        if improved: cur.append(add)
    Pens = avg(cur); s_ens, t_ens = best_thr(Pens, tr)
    # seuils par régime + morphologie (validation croisée 2 moitiés pour éviter le sur-ajustement)
    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]; best = (s_ens, None)
    half = [tr[0::2], tr[1::2]]
    cands = [(tl, th, cut, k) for tl in grid for th in grid for cut in (0.3, 0.5) for k in (0, 1)]
    def sc(names, c): return float(np.mean([miou_image(pred2(Pens, n, *c), gt[n]) for n in names]))
    cv = []
    for a, b in ((0, 1), (1, 0)):
        c = max(cands, key=lambda c: sc(half[a], c)); cv.append(sc(half[b], c))
    c_all = max(cands, key=lambda c: sc(tr, c)); s_reg_cv = float(np.mean(cv)); s_reg_fit = sc(tr, c_all)
    base_cv = float(np.mean([score(Pens, half[b], best_thr(Pens, half[a])[1]) for a, b in ((0, 1), (1, 0))]))
    use_reg = s_reg_cv > base_cv + 0.004
    print(m, f"meilleur seul {single[0][0]:.4f} ({single[0][1]}) | ensemble {cur} {s_ens:.4f} thr {t_ens} | régimes CV {s_reg_cv:.4f} vs seuil unique CV {base_cv:.4f} -> {'régimes ' + str(c_all) if use_reg else 'seuil unique'}", flush=True)
    T = {n: np.mean([load(m, r, 'test')[n] for r in cur], 0) for n in te}
    if use_reg:
        pred = [pred2(T, n, *c_all) for n in te]; sc_new = [miou_image(pred2(Pens, n, *c_all), gt[n]) for n in tr]
    else:
        pred = [T[n] > t_ens for n in te]; sc_new = [miou_image(Pens[n] > t_ens, gt[n]) for n in tr]
    rep[m] = dict(single=[(round(s, 4), r, t) for s, r, t in single], ens=cur, ens_score=s_ens, thr=t_ens, regimes=list(c_all) if use_reg else None,
                  regimes_cv=s_reg_cv, single_thr_cv=base_cv, final_train=float(np.mean(sc_new)))
    names_out += te; masks_out += pred; allsc_new += sc_new
rep["overall_train_oof"] = float(np.mean(allsc_new)); print("OOF GLOBAL attendu:", round(rep["overall_train_oof"], 4))
M = torch.from_numpy(np.stack(masks_out)).bool(); assert M.shape == (1020, 128, 128) and len(set(names_out)) == 1020
torch.save({"image_names": names_out, "masks": M}, f"{O}/sub_orehn_a1.pt"); json.dump(rep, open(f"{O}/sub_orehn_a1.json", "w"), indent=1)
# comparaison avec la dernière soumission assemblée de Bruno si lisible
for p in sorted(glob.glob(f"{B}/wm/s3/out/sub_s3_v*.pt")) + sorted(glob.glob(f"{OUTB}/sub_*.pt")):
    try:
        d = torch.load(p, weights_only=True); old = dict(zip(d["image_names"], d["masks"].numpy().astype(bool))); new = dict(zip(names_out, M.numpy()))
        ag = np.mean([miou_image(new[n], old[n]) for n in names_out]); print(f"accord (mIoU) avec {os.path.basename(p)} : {ag:.3f} ; fraction de pixels à 1 : nouveau {M.float().mean():.3f} / ancien {d['masks'].float().mean():.3f}")
    except Exception as e:
        print("lecture impossible", p, repr(e)[:60])
print("écrit", f"{O}/sub_orehn_a1.pt")
