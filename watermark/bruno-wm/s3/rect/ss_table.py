"""StegaStamp train: one line per not-full image, search result vs ground truth and vs the U-Net (CPU only).
Usage: python rect/ss_table.py <raw json>[,<raw json>...]"""
import json, sys
import numpy as np
from common import *
raw = {}
for f in sys.argv[1].split(","):
    raw.update(json.load(open(f"{OUT}/{f}")))
probe = {r["name"]: r for r in json.load(open(f"{OUT}/rect_ss_probe.json"))}
rows = sorted([(n, r) for n, r in raw.items() if r.get("full") is False], key=lambda t: (t[1]["kind"], -t[1].get("acc", 0)))
for n, r in rows:
    p = probe.get(n, {}); g = r.get("gt_bbox", [0, 0, 0, 0])
    print(f"{n} {r['kind']:5s} gt {g[1] - g[0]:3d}x{g[3] - g[2]:3d} acc@gt {p.get('acc_gt', -1):.2f} unet {p.get('unet', -1):.3f} | "
          + (f"acc hc {r['acc_hc']:.2f} pm {r.get('acc_pm', -1):.2f} pp {r.get('acc_pp', -1):.2f} iou hc {r.get('iou_hc', -1):.2f} pm {r.get('iou_pm', -1):.2f} pp {r.get('iou_pp', -1):.2f} n_hyp {r['n_hyp']} evals {r['n_eval']} t {r['t']:.0f}s" if "rect" in r else f"- (hyp {r.get('n_hyp', 0)})"))
R = [r for _, r in rows if r["kind"] == "rect" and r.get("acc", 0) >= 0.75]
print("plain rectangles accepted at 0.75:", len(R), "mean rect IoU hc %.3f pm %.3f pp %.3f" % tuple(np.mean([r["iou_" + k] for r in R]) for k in ("hc", "pm", "pp")),
      " best-of-three by accuracy tie -> hc")
T = [r["t"] for _, r in rows if "rect" in r]; E = [r["n_eval"] for _, r in rows if "rect" in r]
print("searched", len(T), "mean time %.1fs  median %.1fs  mean decoder calls %.0f" % (np.mean(T), np.median(T), np.mean(E)))
