"""For not-full SHAPE images (object silhouettes, rectangles cut by an object): which frame decodes?
Exhaustive coarse search over rectangles (32-px grid, min side 64) that contain >= 90% of the ground-truth mask,
decoded (a) as is, (b) with everything outside the ground-truth mask greyed. Reports the best bit accuracy and how the
best frame relates to the mask.  usage: frame2.py <method> [limit]"""
import sys, time, numpy as np, torch
from torchvision.ops import roi_align
from common import *
meth = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
wm = load_wm(meth); dev = "cuda"; res = wm.resolution; g = gt128()
edges = list(range(0, 513, 32))
ALL = np.array([(y0, y1, x0, x1) for y0 in edges for y1 in edges if y1 - y0 >= 64 for x0 in edges for x1 in edges if x1 - x0 >= 64
                and max(y1 - y0, x1 - x0) / min(y1 - y0, x1 - x0) <= MAX_ASPECT[meth] * 1.1], np.float32)

@torch.no_grad()
def decode(img, rects, sm, bs=512):
    out_m, out_a = [], []
    for i in range(0, len(rects), bs):
        r = torch.from_numpy(rects[i:i + bs]).to(dev); boxes = torch.stack([torch.zeros(len(r), device=dev), r[:, 2], r[:, 0], r[:, 3], r[:, 1]], 1)
        c = roi_align(img, boxes, output_size=(res, res), sampling_ratio=2, aligned=True)
        if meth == "trustmark":
            with torch.autocast("cuda", dtype=torch.float16): lg = wm.net(c * 2 - 1)
        else: lg = wm.det(c)[:, 1:]
        lg = lg.float(); out_m.append((lg * sm[None]).mean(1)); out_a.append(((lg > 0).float() * 2 - 1 == sm[None]).float().mean(1))
    return torch.cat(out_m).cpu().numpy(), torch.cat(out_a).cpu().numpy()

def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2])); i = iy * ix
    return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)

items = [it for it in load_items("train", meth) if not it["full"] and kind(g[it["name"]]) == "shape"]
if limit: items = items[:limit]
rows = []; t0 = time.time()
for it in items:
    M = gt512(it["name"]); ys, xs = np.nonzero(M); bb = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    I = np.pad(M.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r = ALL.astype(np.int64); inside = I[r[:, 1], r[:, 3]] - I[r[:, 0], r[:, 3]] - I[r[:, 1], r[:, 2]] + I[r[:, 0], r[:, 2]]
    C = np.concatenate([ALL[inside >= 0.9 * M.sum()], np.array([bb], np.float32)])
    x = load_img(it["path"]); sm = sm_of(it["bits"]); m = torch.from_numpy(M).to(dev)[None, None].float()
    out = {}
    for tag, im in [("plain", x), ("keep", x * m + 0.5 * (1 - m))]:
        mg, ac = decode(im, C, sm); k = int(np.argmax(mg)); R = C[k]
        out[tag] = (ac[k], mg[k], iou(R, bb), (R[1] - R[0]) * (R[3] - R[2]) / ((bb[1] - bb[0]) * (bb[3] - bb[2])), M.sum() / ((R[1] - R[0]) * (R[3] - R[2])), ac[-1])
    rows.append((M.mean(), M.sum() / ((bb[1] - bb[0]) * (bb[3] - bb[2])), len(C)) + out["plain"] + out["keep"])
    print(it["name"], f"area {M.mean():.3f} fillbbox {rows[-1][1]:.2f} n={len(C)} | plain: acc {out['plain'][0]:.2f} IoU(best,bbox) {out['plain'][2]:.2f} mask/frame {out['plain'][4]:.2f} | keep: acc {out['keep'][0]:.2f} IoU {out['keep'][2]:.2f} mask/frame {out['keep'][4]:.2f} | at bbox {out['plain'][5]:.2f}/{out['keep'][5]:.2f}", flush=True)
A = np.array(rows); thr = 0.62 if meth == "pixelseal" else 0.72
print(f"\n== {meth} shapes n={len(A)} ({(time.time()-t0)/len(A):.1f}s/img, {A[:,2].mean():.0f} frames/img). Chance-level max over this many frames is about {0.5 + (3.3 if meth != 'pixelseal' else 3.3) * 0.5 / np.sqrt(wm.nbits):.2f}")
for tag, o in [("plain", 3), ("keep ", 9)]:
    ok = A[:, o] > thr
    print(f"   {tag}: best acc mean {A[:,o].mean():.3f} median {np.median(A[:,o]):.3f} | decodable (acc>{thr}) {ok.mean():.2f} | among decodable: IoU(best frame, mask bbox) {A[ok, o+2].mean():.2f}, frame area / bbox area {np.median(A[ok, o+3]):.2f}, mask share of frame {np.median(A[ok, o+4]):.2f}")
    ko = ~ok; print(f"          not decodable n={ko.sum()}: mask area median {np.median(A[ko,0]):.3f}, fill of bbox median {np.median(A[ko,1]):.2f} | decodable n={ok.sum()}: mask area median {np.median(A[ok,0]):.3f}, fill of bbox median {np.median(A[ok,1]):.2f}")
