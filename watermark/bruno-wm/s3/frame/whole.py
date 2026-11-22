"""Whole-image decoder margin and bit accuracy for every image (train + test): the basic 'is the frame the whole
image?' indicator. Saves out/whole_<split>.npz with names, method, margin, acc (+ full, wm_area on train)."""
import numpy as np, torch
from common import *
for split in ["train", "test"]:
    rec = []
    g = gt128() if split == "train" else {}
    for meth in METHODS:
        wm = load_wm(meth)
        for it in load_items(split, meth):
            m, a = signed(wm, load_img(it["path"]), sm_of(it["bits"]))
            rec.append((it["name"], meth, float(m[0]), float(a[0]), -1 if it["full"] is None else int(it["full"]), float(g[it["name"]].mean()) if g else -1.0))
        del wm; torch.cuda.empty_cache()
    n, me, mg, ac, fu, ar = map(np.array, zip(*rec))
    np.savez(f"{OUT}/whole_{split}.npz", names=n, method=me, margin=mg, acc=ac, full=fu, wm_area=ar)
    for meth in METHODS:
        s = me == meth
        if split == "train":
            for thr in [0.62, 0.70, 0.75, 0.80]:
                pred = ac[s] > thr; f = fu[s] == 1
                print(f"{meth:10s} train acc>{thr}: full recall {pred[f].mean():.3f}  not-full flagged {pred[~f].mean():.3f}  (wm area of flagged not-full: {np.round(np.sort(ar[s][~f & pred])[:12], 2)})")
        else:
            print(f"{meth:10s} test  n={s.sum()} share acc>0.75: {(ac[s] > 0.75).mean():.3f}  acc>0.62: {(ac[s] > 0.62).mean():.3f}  acc quantiles {np.round(np.quantile(ac[s], [.1, .25, .5, .75, .9]), 2)}")
