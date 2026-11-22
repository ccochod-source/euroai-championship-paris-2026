"""PixelSeal: re-estimate the edges of the rectangles found by ps_rect.py.
Measured: for a strong watermark the decoder still reads sub-crops of the rectangle (flat margin when an edge moves
INSIDE), and the margin falls when the edge moves OUTSIDE (unwatermarked border enters the crop). So each edge is
set to the outermost position of the plateau: walk outward from the argmax while margin >= level * max.
Usage: sg 2 rect/ps_polish.py --inp rect_raw_pixelseal_train.json --out rect_pol_pixelseal_train.json --levels 0.97,0.93,0.88 --min_acc 0.64"""
import argparse, json, time
import numpy as np, torch
from common import *
import pslib
from rectlib import Scorer, rect_iou, valid

torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


def margin(sc, img, rects, sm):
    out = []
    for i in range(0, len(rects), 128):
        lg = sc.logits_res(sc.crops(img, rects[i:i + 128])); out.append((lg * sm[None]).mean(1))
    return torch.cat(out).cpu().numpy()


def outer(sc, img, sm, rect, level, rounds=3, span_out=0.35, span_in=0.2):
    r = list(rect)
    for _ in range(rounds):
        for e in range(4):
            side = (r[1] - r[0]) if e < 2 else (r[3] - r[2]); sgn = -1 if e in (0, 2) else 1      # outward direction
            step = max(1, int(round(side / 60)))
            offs = list(range(-int(span_in * side / step), int(span_out * side / step) + 1))        # in steps, outward > 0
            cands, keep = [], []
            for o in offs:
                c = list(r); c[e] = r[e] + sgn * o * step
                if valid(tuple(c), min_side=16):
                    cands.append(tuple(c)); keep.append(o)
            if len(cands) < 3:
                continue
            m = margin(sc, img, cands, sm); k = int(np.argmax(m))
            if m[k] <= 0:
                continue
            if MODE == "mid":                                   # midpoint of the plateau around the argmax
                lo = hi = k
                while lo > 0 and m[lo - 1] >= level * m.max(): lo -= 1
                while hi < len(m) - 1 and m[hi + 1] >= level * m.max(): hi += 1
                at_border = lo == 0 or hi == len(m) - 1
                r[e] = cands[k][e] if at_border else int(round((cands[lo][e] + cands[hi][e]) / 2))
            else:
                while k < len(m) - 1 and m[k + 1] >= level * m.max():
                    k += 1
                r[e] = cands[k][e]
    return tuple(r)


ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--levels", default="0.97,0.93,0.88")
ap.add_argument("--min_acc", type=float, default=0.64); ap.add_argument("--mode", default="outer"); ap.add_argument("--span", type=float, default=0.35); ap.add_argument("--split", default="train")
a = ap.parse_args(); MODE = a.mode; TAG = "m" if MODE == "mid" else "o"
wm = pslib.patch(load_wm("pixelseal")); sc = Scorer(wm, amp=False)
raw = json.load(open(f"{OUT}/{a.inp}")); items = {it["name"]: it for it in load_items(a.split, "pixelseal")}
levels = [float(v) for v in a.levels.split(",")]; t0 = time.time(); n = 0
for name, r in raw.items():
    if r.get("acc", 0) < a.min_acc:
        continue
    it = items[name]; img = load_img(it["path"]); sm = sm_of(it["bits"]); n += 1
    line = f"{name} {r.get('kind')} acc {r['acc']:.2f} mar {r.get('mar', 0):.2f} iou hc {r.get('rect_iou_hc', -1):.2f} pm {r.get('rect_iou', -1):.2f}"
    for lv in levels:
        q = outer(sc, img, sm, r["rect"], lv, span_out=a.span, span_in=a.span if MODE == "mid" else 0.2); s, acc = sc.score(img, [q], sm); tag = f"{TAG}{int(round(lv * 100))}"
        r["rect_" + tag] = [int(v) for v in q]; r["acc_" + tag] = float(acc[0]); r["soft_" + tag] = float(s[0])
        if "gt_bbox" in r:
            r["iou_" + tag] = rect_iou(q, r["gt_bbox"]); line += f" {tag} {r['iou_' + tag]:.2f} (acc {acc[0]:.2f})"
    print(line, flush=True)
json.dump(raw, open(f"{OUT}/{a.out}", "w"))
print("DONE", n, "rects", f"{time.time() - t0:.0f}s")
if a.split == "train":
    R = [r for r in raw.values() if r.get("kind") == "rect" and not r.get("full") and r.get("acc", 0) >= 0.7]
    print("train plain rectangles with acc >= 0.7: n", len(R), " mean rect IoU: hc %.3f pm %.3f" % (np.mean([r["rect_iou_hc"] for r in R]), np.mean([r["rect_iou"] for r in R])),
          " ".join(f"{TAG}{int(round(lv * 100))} {np.mean([r['iou_%s%d' % (TAG, int(round(lv * 100)))] for r in R]):.3f}" for lv in levels))
