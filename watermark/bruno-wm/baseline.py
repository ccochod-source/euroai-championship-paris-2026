"""Image-level baseline: all-ones if the global decoder reads the known message, else all-zeros."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/p/scratch/training2666/dasilvalopes3/wm/kit")
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor, score
R = "/p/scratch/training2666/dasilvalopes3/wm/release"; OUT = "/p/scratch/training2666/dasilvalopes3/wm/out"
import os; os.makedirs(OUT, exist_ok=True)
res = {}
for split in ["train", "test"]:
    for meth in ["trustmark", "stegastamp", "pixelseal"]:
        wm = load_watermark(meth, device="cuda")
        ds = LocalisationDataset(R, split, methods=[meth])
        for it in ds:
            lg = wm.logits(to_tensor(it["image"]))[0].cpu().numpy()
            acc = float(((lg > 0).astype(int) == it["message"]).mean())
            res[it["image_name"]] = dict(split=split, method=meth, acc=acc, full=bool(it.get("full_watermark", False)), gt=it.get("mask128"))
        del wm
tr = [v for v in res.values() if v["split"] == "train"]
for meth in ["trustmark", "stegastamp", "pixelseal"]:
    a = np.array([v["acc"] for v in tr if v["method"] == meth]); f = np.array([v["full"] for v in tr if v["method"] == meth])
    print(meth, "acc full: min %.3f mean %.3f | notfull: max %.3f mean %.3f" % (a[f].min(), a[f].mean(), a[~f].max(), a[~f].mean()))
THR = 0.75
pred = lambda v: np.full((128, 128), int(v["acc"] > THR), np.uint8)
print("train score baseline:", score([pred(v) for v in tr], [v["gt"] for v in tr]))
te = {k: v for k, v in res.items() if v["split"] == "test"}
print("test: predicted full fraction", np.mean([v["acc"] > THR for v in te.values()]))
names = list(te); masks = torch.from_numpy(np.stack([pred(te[n]) for n in names])).bool()
torch.save({"image_names": names, "masks": masks}, OUT + "/sub_baseline.pt")
json.dump({k: dict(method=v["method"], split=v["split"], acc=v["acc"]) for k, v in res.items()}, open(OUT + "/global_acc.json", "w"))
print("BASELINE_DONE", len(names))
