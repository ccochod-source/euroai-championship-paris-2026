"""PixelSeal: find the watermarked rectangle of a not-full image with the official decoder.
Stage 1 (dense, convolutional): the detector is slid over the image resized anisotropically so that one final cell
  = cy x cx image pixels, with LOCAL response normalisation (pslib). A rectangle decodes when it covers about
  8-13 cells per side; window logits = mean of the per-position logits over kh x kw cells. The same windows are scored
  against D random decoy messages, which gives a per-image null distribution of the best window.
Stage 2 (exact): the best windows are re-scored on crops with the original detector, moved by hill climbing, each
  edge is then polished on the raw margin.
Writes raw results (every searched image, no threshold) to <out>; finalize.py applies the acceptance rule.
Usage: sg 2 rect/ps_rect.py --split train --out rect_raw_pixelseal_train.json [--decoy 1] [--limit n] [--only_notfull 1]"""
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F
from common import *
import pslib
from rectlib import Scorer, refine, rect_iou, valid

torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
KS = (8, 9, 10, 11, 12, 13)


def cell_sizes(cmin=4.5, ratio=1.3):
    out = []; c = cmin
    while c * KS[-1] < 512 * ratio:
        out.append(c); c *= ratio
    return out


@torch.no_grad()
def stage1(wm, img, sm, decoys, cys, k_grn=9, max_aspect=16.0):
    """Returns (candidates sorted by accuracy: list of (acc, rect)), best accuracy, per-decoy best accuracy."""
    cands, null = [], torch.zeros(decoys.shape[1], device=img.device)
    for cy in cys:
        Hf = max(8, int(round(512 / cy)))
        for cx in cys:
            Wf = max(8, int(round(512 / cx)))
            x = F.interpolate(img, size=(Hf * 32, Wf * 32), mode="bilinear", align_corners=False, antialias=True)
            L = pslib.ppl(wm, x, (k_grn, k_grn), amp=False)[0].double()
            I = F.pad(L.cumsum(1).cumsum(2), (1, 0, 1, 0))
            py, px = 512 / Hf, 512 / Wf
            for kh in KS:
                if kh > Hf:
                    continue
                for kw in KS:
                    if kw > Wf:
                        continue
                    h, w = kh * py, kw * px
                    if max(h, w) / min(h, w) > max_aspect * 1.1:
                        continue
                    S = I[:, kh:, kw:] - I[:, :-kh, kw:] - I[:, kh:, :-kw] + I[:, :-kh, :-kw]
                    sg = (S > 0).float() * 2 - 1                                        # 256 x Ph x Pw
                    acc = (sg * sm[:, None, None]).mean(0) * 0.5 + 0.5
                    null = torch.maximum(null, (torch.einsum("bp,bd->pd", sg.flatten(1), decoys) / sg.shape[0]).amax(0) * 0.5 + 0.5)
                    v, idx = acc.flatten().topk(min(2, acc.numel()))
                    for vv, ii in zip(v.tolist(), idx.tolist()):
                        i, j = divmod(ii, acc.shape[1])
                        cands.append((vv, (i * py, (i + kh) * py, j * px, (j + kw) * px)))
    cands.sort(key=lambda t: -t[0])
    return cands, cands[0][0], null.cpu().numpy()


def clip_rect(r):
    y0, y1, x0, x1 = [int(round(v)) for v in r]
    y0, x0 = max(0, min(y0, 504)), max(0, min(x0, 504)); y1, x1 = min(512, max(y1, y0 + 8)), min(512, max(x1, x0 + 8))
    return (y0, y1, x0, x1)


def nms(cands, k, thr=0.5):
    out = []
    for v, r in cands:
        r = clip_rect(r)
        if all(rect_iou(r, q) < thr for _, q in out):
            out.append((v, r))
        if len(out) >= k:
            break
    return out


def margin(sc, img, rects, sm):
    out = []
    for i in range(0, len(rects), 128):
        lg = sc.logits_res(sc.crops(img, rects[i:i + 128])); out.append((lg * sm[None]).mean(1))
    return torch.cat(out).cpu().numpy()


def jitter(r, d):
    """81 rectangles: every edge moved by -d, 0 or +d (d: fraction of the side)."""
    y0, y1, x0, x1 = r; dy, dx = max(1, int(round(d * (y1 - y0)))), max(1, int(round(d * (x1 - x0))))
    out = []
    for a in (-dy, 0, dy):
        for b in (-dy, 0, dy):
            for c in (-dx, 0, dx):
                for e in (-dx, 0, dx):
                    q = (y0 + a, y1 + b, x0 + c, x1 + e)
                    if valid(q, min_side=16):
                        out.append(q)
    return out


def polish(sc, img, sm, rect, rounds=2, span=0.2):
    """Per-edge scan of the raw margin, argmax."""
    r = list(rect)
    for _ in range(rounds):
        for e in range(4):
            side = (r[1] - r[0]) if e < 2 else (r[3] - r[2])
            step = max(1, int(round(side / 60))); n = max(2, int(span * side / step))
            cands, keep = [], []
            for o in [k * step for k in range(-n, n + 1)]:
                c = list(r); c[e] = r[e] + o
                if valid(tuple(c), min_side=16):
                    cands.append(tuple(c)); keep.append(o)
            if len(cands) < 3:
                continue
            m = margin(sc, img, cands, sm); r[e] = r[e] + keep[int(np.argmax(m))]
    return tuple(r)


def stage2(sc, img, sm, cands, topk=3):
    best = None
    for v, r in nms(cands, topk):
        J = jitter(r, 0.05); s, a = sc.score(img, J, sm, bs=128); r0 = J[int(np.argmax(s))]
        rr, s, acc = refine(sc, img, sm, r0, max_aspect=16.0, first_step=max(1, int(round(min(r0[1] - r0[0], r0[3] - r0[2]) * 0.06))))
        if best is None or s > best[1]:
            best = (rr, s, acc, v, r)
    m0 = float(margin(sc, img, [best[0]], sm)[0])
    rr = polish(sc, img, sm, best[0], rounds=3, span=0.3)
    m1 = float(margin(sc, img, [rr], sm)[0])
    if m1 < m0:
        rr, m1 = best[0], m0
    s, acc = sc.score(img, [rr], sm)
    return dict(rect=[int(v) for v in rr], soft=float(s[0]), acc=float(acc[0]), mar=m1, rect_hc=[int(v) for v in best[0]],
                acc_hc=float(best[2]), soft_hc=float(best[1]), mar_hc=m0, rect_conv=[int(v) for v in best[4]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decoy", type=int, default=0); ap.add_argument("--only_notfull", type=int, default=0)
    ap.add_argument("--names", default=""); ap.add_argument("--shard", default="0/1"); ap.add_argument("--resume", type=int, default=0); ap.add_argument("--p_gate", type=float, default=0.3)
    a = ap.parse_args()
    wm = pslib.patch(load_wm("pixelseal")); sc = Scorer(wm, amp=False)
    z = np.load(f"{OUT}/whole_{a.split}.npz"); whole = {str(n): float(v) for n, v in zip(z["names"], z["acc"])}
    items = load_items(a.split, "pixelseal")
    if a.only_notfull:
        items = [it for it in items if it["full"] is False]
    if a.names:
        items = [it for it in items if it["name"] in a.names.split(",")]
    if a.limit:
        items = items[:a.limit]
    k0, kn = [int(v) for v in a.shard.split("/")]
    items = [it for it in items if whole[it["name"]] < 0.8][k0::kn] + ([it for it in items if whole[it["name"]] >= 0.8] if k0 == 0 else [])
    G = gt128() if a.split == "train" else {}
    g = torch.Generator(device="cuda").manual_seed(1234)
    decoys = (torch.rand(256, 64, device="cuda", generator=g) > 0.5).float() * 2 - 1
    cys = cell_sizes(); print("cell sizes", [round(c, 1) for c in cys], flush=True)
    res, t0 = {}, time.time()
    if a.resume and os.path.exists(f"{OUT}/{a.out}"):
        res = json.load(open(f"{OUT}/{a.out}")); print("resuming with", len(res), "images done", flush=True)
    for n, it in enumerate(items):
        if it["name"] in res:
            continue
        r = dict(acc_whole=whole[it["name"]])
        if whole[it["name"]] < 0.8:
            img = load_img(it["path"]); sm = sm_of(it["bits"])
            if a.decoy:
                gg = torch.Generator(device="cuda").manual_seed(a.decoy * 100003 + n)
                sm = (torch.rand(256, device="cuda", generator=gg) > 0.5).float() * 2 - 1
            t1 = time.time()
            cands, best, null = stage1(wm, img, sm, decoys, cys)
            r.update(conv=best, null_mean=float(null.mean()), null_std=float(null.std()), null_max=float(null.max()),
                     p=float((null >= best).mean()), t1=time.time() - t1)
            r["z"] = (best - r["null_mean"]) / max(r["null_std"], 1e-3)
            if r["p"] <= a.p_gate:
                t2 = time.time(); r.update(stage2(sc, img, sm, cands)); r["t2"] = time.time() - t2
        if it["name"] in G:
            gt = G[it["name"]]; r["kind"] = kind(gt); r["full"] = bool(it["full"])
            m5 = gt512(it["name"]); ys, xs = np.nonzero(m5)
            if len(ys):
                r["gt_bbox"] = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]
                if "rect" in r:
                    r["rect_iou"] = rect_iou(r["rect"], r["gt_bbox"]); r["rect_iou_hc"] = rect_iou(r["rect_hc"], r["gt_bbox"])
        res[it["name"]] = r
        print(n, len(items), f"{time.time() - t0:.0f}s", it["name"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if n % 10 == 0:
            json.dump(res, open(f"{OUT}/{a.out}", "w"))
    json.dump(res, open(f"{OUT}/{a.out}", "w"))
    print("DONE", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
