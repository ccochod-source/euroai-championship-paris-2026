"""Bank of realistic object shapes from COCO val2017 instance masks (tight crops, max side 128, bool)."""
import json, pickle, numpy as np, cv2
from pycocotools import mask as mu
from common import CLEAN, SYN
d = json.load(open(f"{CLEAN}/annotations/instances_val2017.json"))
imgs = {im["id"]: im for im in d["images"]}
bank = []
for a in d["annotations"]:
    if a.get("iscrowd") or a["area"] < 1500: continue
    im = imgs[a["image_id"]]; seg = a["segmentation"]
    rle = mu.merge(mu.frPyObjects(seg, im["height"], im["width"])) if isinstance(seg, list) else seg
    m = mu.decode(rle).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(ys) == 0: continue
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = 128 / max(m.shape)
    m = cv2.resize(m, (max(2, round(m.shape[1] * s)), max(2, round(m.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    bank.append(m > 0)
    if len(bank) >= 25000: break
import os; os.makedirs(SYN, exist_ok=True)
pickle.dump(bank, open(f"{SYN}/shape_bank.pkl", "wb"))
print("MASKS_DONE", len(bank))
