"""Print one line per searched PixelSeal train image (CPU only): evidence of both stages vs ground truth."""
import json, sys
import numpy as np
from common import *
raw = json.load(open(f"{OUT}/{sys.argv[1]}"))
rows = [(n, r) for n, r in raw.items() if "conv" in r]
rows.sort(key=lambda t: (t[1].get("full", False), t[1].get("kind", "?"), -t[1].get("acc", 0)))
for n, r in rows:
    g = r.get("gt_bbox", [0, 0, 0, 0])
    print(f"{n} {'FULL' if r.get('full') else 'part'} {r.get('kind', '?'):5s} gt {g[1] - g[0]:3d}x{g[3] - g[2]:3d} whole {r['acc_whole']:.2f} conv {r['conv']:.3f} z {r['z']:5.1f} p {r['p']:.2f} | "
          + (f"acc {r['acc']:.3f} hc {r['acc_hc']:.3f} mar {r['mar']:.2f} soft {r['soft']:.2f} iou {r.get('rect_iou', -1):.2f} hc {r.get('rect_iou_hc', -1):.2f} rect {r['rect']}" if "rect" in r else "-"))
S = [r for _, r in rows if "rect" in r]
print("searched", len(rows), "stage 2 run on", len(S))
