"""StegaStamp probe: speed of the ONNX decoder, decodability at the ground-truth rectangle, quality of the U-Net
bounding box as a starting point (edge errors as a fraction of the side), whole-image stats."""
import time, json
import numpy as np, torch, torch.nn.functional as F
from common import *
from rectlib import Scorer, rect_iou

wm = load_wm("stegastamp"); sc = Scorer(wm)
print("providers", wm.sess.get_providers())
x = torch.rand(64, 3, 400, 400, device="cuda")
for bs in [32, 64]:
    sc.logits_res(x[:bs]); t = time.time(); sc.logits_res(x[:bs]); sc.logits_res(x[:bs]); print("bs", bs, "crops/s", 2 * bs / (time.time() - t))
z = np.load(f"{W}/wm/out/stegastamp_ftA_trainprobs.npz"); P = {str(n): p for n, p in zip(z["names"], z["probs"])}
print("probs", z["probs"].shape, z["probs"].dtype, float(z["probs"].min()), float(z["probs"].max()))
zw = np.load(f"{OUT}/whole_train.npz"); whole = {str(n): float(v) for n, v in zip(zw["names"], zw["acc"])}
items = [it for it in load_items("train", "stegastamp")]
G = gt128(); rows = []
nf = [it for it in items if not it["full"]]
print("n", len(items), "not full", len(nf), "full with whole<0.8:", sum(whole[it["name"]] < 0.8 for it in items if it["full"]), "not-full with whole>=0.8:", sum(whole[it["name"]] >= 0.8 for it in nf))
for it in nf:
    gt = G[it["name"]]; k = kind(gt); p = P[it["name"]].astype(np.float32)
    if p.max() > 1.5:
        p = p / 255.0
    pm = p > 0.5
    r = dict(name=it["name"], kind=k, unet=miou(pm, gt), zeros=miou(np.zeros_like(gt), gt))
    m = gt512(it["name"]); ys, xs = np.nonzero(m); g = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    img = load_img(it["path"]); sm = sm_of(it["bits"])
    s, a = sc.score(img, [g], sm); r.update(gt_bbox=g, acc_gt=float(a[0]), soft_gt=float(s[0]))
    if pm.any():
        ys, xs = np.nonzero(pm); u = (int(ys.min()) * 4, int(ys.max()) * 4 + 4, int(xs.min()) * 4, int(xs.max()) * 4 + 4)
        s, a = sc.score(img, [u], sm); h, w = g[1] - g[0], g[3] - g[2]
        r.update(unet_bbox=u, acc_unet=float(a[0]), iou_bbox=rect_iou(u, g), err=[(u[0] - g[0]) / h, (u[1] - g[1]) / h, (u[2] - g[2]) / w, (u[3] - g[3]) / w])
    rows.append(r)
    print({k: ([round(float(e), 2) for e in v] if k == "err" else round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(rows, open(f"{OUT}/rect_ss_probe.json", "w"))
for k in ["rect", "shape"]:
    R = [r for r in rows if r["kind"] == k]
    print(k, "n", len(R), "zeros %.3f unet %.3f" % (np.mean([r["zeros"] for r in R]), np.mean([r["unet"] for r in R])),
          "acc_gt mean %.3f >0.75: %.2f" % (np.mean([r["acc_gt"] for r in R]), np.mean([r["acc_gt"] > .75 for r in R])),
          "unet has bbox %.2f" % np.mean(["unet_bbox" in r for r in R]),
          "acc at unet bbox >0.75: %.2f" % np.mean([r.get("acc_unet", 0) > .75 for r in R]),
          "max |edge err| <0.04: %.2f <0.08: %.2f <0.15: %.2f" % tuple(np.mean([("err" in r and np.abs(r["err"]).max() < t) for r in R]) for t in (0.04, 0.08, 0.15)))
