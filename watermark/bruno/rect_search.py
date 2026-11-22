"""Find the watermarked rectangle of a partially watermarked image by decoding the known message.
Stage 1 (TrustMark): dense multi-scale search, the decoder is run convolutionally on the image resized so that a
          candidate h x w rectangle maps to the decoder's working resolution; window logits = average of the
          per-position logits over an 8 x 8 feature window.
Stage 2: exact scoring of the best candidates on crops, hill climbing, then per-edge polishing on the raw margin.
Usage: python rect_search.py --kit . --data release --method trustmark --split train --out bruno/rects_trustmark_train.npz
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from rectlib import Scorer, refine, rect_iou, valid  # noqa

MAXA = {"pixelseal": 16.0, "stegastamp": 16.0, "trustmark": 2.0}


def trunk_tm(net, x):
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    return net.layer4(net.layer3(net.layer2(net.layer1(x))))


@torch.no_grad()
def conv_candidates_tm(wm, img, sm, sizes, max_aspect, per_pair=2):
    net, res = wm.net, wm.resolution
    Wt, b = net.fc.weight.float(), net.fc.bias.float()
    out = []
    for h in sizes:
        for w in sizes:
            if max(h, w) / min(h, w) > max_aspect * 1.05:
                continue
            Hr, Wr = max(res, int(round(512 * res / h))), max(res, int(round(512 * res / w)))
            x = F.interpolate(img, size=(Hr, Wr), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
            with torch.autocast("cuda", dtype=torch.float16):
                f = trunk_tm(net, x)
            pp = F.conv2d(f.float(), Wt[:, :, None, None], b)
            kh, kw = min(8, pp.shape[2]), min(8, pp.shape[3])
            L = F.avg_pool2d(pp, (kh, kw), stride=1)
            agr = (torch.tanh(L / 2) * sm[None, :, None, None]).mean(1)[0]
            v, idx = agr.flatten().topk(min(per_pair, agr.numel()))
            sy, sx = 512 / Hr, 512 / Wr
            for vv, ii in zip(v.tolist(), idx.tolist()):
                i, j = divmod(ii, agr.shape[1])
                cy, cx = (32 * i + 16 * kh) * sy, (32 * j + 16 * kw) * sx
                out.append((vv, (cy - h / 2, cy + h / 2, cx - w / 2, cx + w / 2)))
    return sorted(out, reverse=True)


def clip_rect(r):
    y0, y1, x0, x1 = [int(round(v)) for v in r]
    y0, x0 = max(0, min(y0, 504)), max(0, min(x0, 504)); y1, x1 = min(512, max(y1, y0 + 8)), min(512, max(x1, x0 + 8))
    return (y0, y1, x0, x1)


def margin(sc, img, rects, sm):
    out = []
    for i in range(0, len(rects), 256):
        lg = sc.logits_res(sc.crops(img, rects[i:i + 256])); out.append((lg * sm[None]).mean(1))
    return torch.cat(out).cpu().numpy()


def polish(sc, img, sm, rect, plateau, rounds=2, span=0.25, level=0.8):
    """Per-edge scan of the raw margin. plateau=True: midpoint of the region above level * max (flat objective),
    else argmax."""
    r = list(rect)
    for _ in range(rounds):
        for e in range(4):
            side = (r[1] - r[0]) if e < 2 else (r[3] - r[2])
            step = max(1, int(round(side / 60))); n = max(2, int(span * side / step))
            offs = [k * step for k in range(-n, n + 1)]
            cands, keep = [], []
            for o in offs:
                c = list(r); c[e] = r[e] + o
                if valid(tuple(c)):
                    cands.append(tuple(c)); keep.append(o)
            if len(cands) < 3:
                continue
            m = margin(sc, img, cands, sm); k = int(np.argmax(m))
            if plateau and m[k] > 0:
                lo = hi = k
                while lo > 0 and m[lo - 1] >= level * m[k]: lo -= 1
                while hi < len(m) - 1 and m[hi + 1] >= level * m[k]: hi += 1
                at_border = (lo == 0 or hi == len(m) - 1)
                r[e] = r[e] + (keep[k] if at_border else int(round((keep[lo] + keep[hi]) / 2)))
            else:
                r[e] = r[e] + keep[k]
    return tuple(r)


def nms(cands, k, thr=0.6):
    out = []
    for v, r in cands:
        r = clip_rect(r)
        if all(rect_iou(r, q) < thr for _, q in out):
            out.append((v, r))
        if len(out) >= k:
            break
    return out


def search(wm, sc, img, sm, sizes, topk=6):
    meth = wm.name
    cands = nms(conv_candidates_tm(wm, img, sm, sizes, MAXA[meth]), topk)
    best = None
    for v, r in cands:
        rr, s, acc = refine(sc, img, sm, r, max_aspect=MAXA[meth])
        if best is None or s > best[1]:
            best = (rr, s, acc, v)
    rr = polish(sc, img, sm, best[0], plateau=(meth == "trustmark"))
    s, acc = sc.score(img, [rr], sm)
    return dict(rect=[int(v) for v in rr], soft=float(s[0]), acc=float(acc[0]), conv=float(best[3]), rect_hc=[int(v) for v in best[0]], acc_hc=float(best[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True); ap.add_argument("--data", required=True); ap.add_argument("--method", default="trustmark")
    ap.add_argument("--split", default="train"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--acc_thr", type=float, default=0.75); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    sys.path.insert(0, a.kit)
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import miou_image, downsample_mask
    wm = load_watermark(a.method, models_dir=Path(a.kit) / "models", device="cuda"); sc = Scorer(wm)
    ds = LocalisationDataset(a.data, a.split, methods=[a.method])
    sizes = [int(round(32 * 1.18 ** k)) for k in range(17)]; sizes = [s for s in sizes if s <= 512] + [512]
    k0, kn = [int(v) for v in a.shard.split("/")]
    res, t0 = {}, time.time()
    idx = list(range(len(ds)))[k0::kn]
    if a.limit:
        idx = idx[:a.limit]
    for n, i in enumerate(idx):
        it = ds[i]
        img = torch.from_numpy(np.array(it["image"])).permute(2, 0, 1)[None].float().div(255).cuda()
        sm = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).cuda()
        _, acc_whole = sc.score(img, [(0, 512, 0, 512)], sm)
        r = dict(acc_whole=float(acc_whole[0]), full=bool(it.get("full_watermark", False)))
        if acc_whole[0] <= a.acc_thr:
            r.update(search(wm, sc, img, sm, sizes))
        if "mask" in it:
            mask = it["mask"]; ys, xs = np.nonzero(mask)
            if len(ys):
                g = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
                r["gt_bbox"] = g; r["fill"] = float(mask[g[0]:g[1], g[2]:g[3]].mean())
            pred = np.zeros((512, 512), np.float32)
            if r.get("acc", 0) >= 0.85:
                y0, y1, x0, x1 = r["rect"]; pred[y0:y1, x0:x1] = 1
            r["miou"] = float(miou_image(downsample_mask(pred), it["mask128"])); r["miou_zeros"] = float(miou_image(np.zeros((128, 128)), it["mask128"]))
            if "rect" in r and "gt_bbox" in r:
                r["rect_iou"] = rect_iou(r["rect"], r["gt_bbox"]); r["rect_iou_hc"] = rect_iou(r["rect_hc"], r["gt_bbox"])
        res[it["image_name"]] = r
        if n % 10 == 0:
            print(n, len(idx), f"{time.time() - t0:.0f}s", it["image_name"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(res, open(a.out, "w"))
    R = [r for r in res.values() if not r["full"]] if a.split == "train" else []
    if R:
        plain = [r for r in R if r.get("fill", 0) > 0.98]; obj = [r for r in R if r.get("fill", 0) <= 0.98]
        f = lambda S, k: float(np.mean([r.get(k, 0) for r in S])) if S else float("nan")
        print(f"\n== {a.method} train, not-full images: n={len(R)}  mIoU zeros={f(R, 'miou_zeros'):.3f} -> rect search={f(R, 'miou'):.3f}")
        print(f"   plain rectangles n={len(plain)}: found (acc>=0.85) {np.mean([r.get('acc', 0) >= .85 for r in plain]):.2f}  mIoU {f(plain, 'miou_zeros'):.3f} -> {f(plain, 'miou'):.3f}  "
              f"rect IoU (found): polished {np.mean([r['rect_iou'] for r in plain if r.get('acc', 0) >= .85]):.3f} / hill-climb only {np.mean([r['rect_iou_hc'] for r in plain if r.get('acc', 0) >= .85]):.3f}")
        print(f"   objects n={len(obj)}: false rectangles (acc>=0.85) {np.mean([r.get('acc', 0) >= .85 for r in obj]):.2f}  mIoU {f(obj, 'miou_zeros'):.3f} -> {f(obj, 'miou'):.3f}")
    print("DONE", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
