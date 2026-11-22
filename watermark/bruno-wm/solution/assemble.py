"""Build a submission from per-method probability files (out/{method}_{run}_trainprobs.npz / _testprobs.npz) or from
the image-level baseline. One threshold per method, tuned on the out-of-fold train probabilities.
usage: assemble.py OUTNAME pixelseal=RUN trustmark=RUN stegastamp=RUN   (RUN may be 'baseline', or 'runA+runB' to average)"""
import sys, json, numpy as np, torch
from common import OUT, RELEASE, METHODS
from wmloc.utils import miou_image

def load(method, run, split):
    acc = None
    for r in run.split("+"):
        z = np.load(f"{OUT}/{method}_{r}_{split}probs.npz"); d = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
        acc = d if acc is None else {k: acc[k] + d[k] for k in acc}
    n = len(run.split("+")); return {k: v / n for k, v in acc.items()}

name = sys.argv[1]; src = dict(a.split("=") for a in sys.argv[2:])
gacc = json.load(open(f"{OUT}/global_acc.json"))
gt = {}; z = np.load(f"{RELEASE}/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
meta = {s: json.load(open(f"{RELEASE}/{s}/metadata.json")) for s in ("train", "test")}
names, masks, rep, allsc = [], [], {}, []
for m in METHODS:
    tr = [x["image_name"] for x in meta["train"] if x["method"] == m]; te = [x["image_name"] for x in meta["test"] if x["method"] == m]
    full = np.array([x["full_watermark"] for x in meta["train"] if x["method"] == m])
    if src.get(m, "baseline") == "baseline":
        f = lambda n: np.full((128, 128), gacc[n]["acc"] > 0.75, bool); thr = None
        sc = np.array([miou_image(f(n), gt[n]) for n in tr]); pred = [f(n) for n in te]
    else:
        P = load(m, src[m], "train"); T = load(m, src[m], "test")
        grid = np.round(np.arange(0.1, 0.91, 0.05), 2)
        best = max(grid, key=lambda t: np.mean([miou_image(P[n] > t, gt[n]) for n in tr])); thr = float(best)
        sc = np.array([miou_image(P[n] > thr, gt[n]) for n in tr]); pred = [T[n] > thr for n in te]
    rep[m] = dict(src=src.get(m, "baseline"), thr=thr, train=float(sc.mean()), full=float(sc[full].mean()), notfull=float(sc[~full].mean()))
    print(m, rep[m]); names += te; masks += pred; allsc.append(sc)
rep["overall_train"] = float(np.concatenate(allsc).mean()); print("EXPECTED (train, out-of-fold) mIoU:", round(rep["overall_train"], 4))
M = torch.from_numpy(np.stack(masks)).bool(); assert M.shape == (1020, 128, 128) and len(set(names)) == 1020
torch.save({"image_names": names, "masks": M}, f"{OUT}/sub_{name}.pt"); json.dump(rep, open(f"{OUT}/sub_{name}.json", "w"), indent=1)
print("wrote", f"{OUT}/sub_{name}.pt")
