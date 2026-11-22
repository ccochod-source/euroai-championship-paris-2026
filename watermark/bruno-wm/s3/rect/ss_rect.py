"""StegaStamp: find the watermarked rectangle, starting from the U-Net probability map of another session and
refining with the official decoder, which only reads the message on (almost) the exact embedding frame.
1. Edge hypotheses from the connected components of the U-Net map (bounding box, 'core' box from the row/column
   profiles, box of the low-threshold component) -> all combinations are decoded, the two best kept as starts.
2. Local search on the soft agreement: +-4 px jitter of the four edges (81), wide scan of each edge in turn, hill
   climbing; if 0.62 <= accuracy < 0.9 a joint scan of opposite edges, and (--rand n) n random perturbations of the
   four edges around the starts, each followed by edge scans and hill climbing.
3. Margin polish variants are stored (rect_pm, rect_pp) but did not beat the hill-climbing result (rect_hc = rect).
The decoder is the decoder-only sub-graph written by ss_make_dec.py (identical logits, GPU memory capped).
Writes raw results (no threshold) to s3/out/<out>; finalize.py applies the acceptance rule and the 3 px shrink.
Usage: rect/sg8 2 rect/ss_rect.py --split train --out rect_raw_stegastamp_train.json
       rect/sg8 2 rect/ss_rect.py --split train --out rect_raw2_stegastamp_train.json --prev rect_raw_stegastamp_train.json --rand 256"""
import argparse, json, time, itertools
import numpy as np, torch
from scipy import ndimage
from common import *
from rectlib import Scorer, refine, rect_iou, valid


def comps(p, thr):
    lab, n = ndimage.label(p > thr)
    out = []
    for k in range(1, n + 1):
        m = lab == k
        out.append((float(p[m].sum()), m))
    return [m for _, m in sorted(out, key=lambda t: -t[0])]


def box(m):
    ys, xs = np.nonzero(m)
    return [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]


def core(m, frac=0.5):
    r, c = m.sum(1), m.sum(0)
    ys, xs = np.nonzero(r >= frac * r.max())[0], np.nonzero(c >= frac * c.max())[0]
    return [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]


def hypotheses(p, max_comp=2):
    """List of rectangles (512-px coordinates) built from per-edge hypotheses."""
    C = comps(p, 0.5); lowC = comps(p, 0.2)
    if not C:
        C = lowC
    rects = []
    for m in C[:max_comp]:
        if m.sum() < 4:
            continue
        hyp = [box(m), core(m, 0.5), core(m, 0.8)]
        for lm in lowC:
            if (lm & m).any():
                hyp.append(box(lm)); break
        edges = [sorted(set(h[e] for h in hyp)) for e in range(4)]
        for y0, y1, x0, x1 in itertools.product(*edges):
            r = (y0 * 4, y1 * 4, x0 * 4, x1 * 4)
            if valid(r, min_side=8) and r not in rects:
                rects.append(r)
    if (p > 0.5).any():
        b = box(p > 0.5); r = (b[0] * 4, b[1] * 4, b[2] * 4, b[3] * 4)
        if valid(r, min_side=8) and r not in rects:
            rects.append(r)
    return rects


def jitter(r, d=4):
    out = []
    for a, b, c, e in itertools.product((-d, 0, d), repeat=4):
        q = (r[0] + a, r[1] + b, r[2] + c, r[3] + e)
        if valid(q, min_side=8):
            out.append(q)
    return out


def edge_scan(sc, img, sm, rect, span=0.4, rel_step=0.025, rounds=2):
    r = list(rect); best = None
    for _ in range(rounds):
        for e in range(4):
            side = (r[1] - r[0]) if e < 2 else (r[3] - r[2])
            step = max(1, int(round(side * rel_step))); n = max(3, int(span * side / step))
            cands, keep = [], []
            for o in [k * step for k in range(-n, n + 1)]:
                c = list(r); c[e] = r[e] + o
                if valid(tuple(c), min_side=8):
                    cands.append(tuple(c)); keep.append(o)
            s, a = sc.score(img, cands, sm, bs=64); k = int(np.argmax(s))
            if best is None or s[k] > best:
                best = float(s[k]); r[e] = r[e] + keep[k]
    return tuple(r)


def margin(sc, img, rects, sm):
    out = []
    for i in range(0, len(rects), 64):
        lg = sc.logits_res(sc.crops(img, rects[i:i + 64])); out.append((lg * sm[None]).mean(1))
    return torch.cat(out).cpu().numpy()


def polish(sc, img, sm, rect, plateau, rounds=2, span=0.12, level=0.8):
    """Per-edge scan of the raw margin (unsaturated). plateau=True: midpoint of the region above level * max,
    else argmax."""
    r = list(rect)
    for _ in range(rounds):
        for e in range(4):
            side = (r[1] - r[0]) if e < 2 else (r[3] - r[2])
            step = max(1, int(round(side / 80))); n = max(3, int(span * side / step))
            cands, keep = [], []
            for o in [k * step for k in range(-n, n + 1)]:
                c = list(r); c[e] = r[e] + o
                if valid(tuple(c), min_side=8):
                    cands.append(tuple(c)); keep.append(o)
            if len(cands) < 3:
                continue
            m = margin(sc, img, cands, sm); k = int(np.argmax(m))
            if plateau and m[k] > 0:
                lo = hi = k
                while lo > 0 and m[lo - 1] >= level * m[k]: lo -= 1
                while hi < len(m) - 1 and m[hi + 1] >= level * m[k]: hi += 1
                r[e] = r[e] + int(round((keep[lo] + keep[hi]) / 2))
            else:
                r[e] = r[e] + keep[k]
    return tuple(r)


def pair_scan(sc, img, sm, rect, span=0.3, rel_step=0.05):
    """Joint scan of two opposite edges (for starts where both are wrong): (x0, x1), then (y0, y1)."""
    r = list(rect)
    for e0, e1 in ((2, 3), (0, 1)):
        side = r[e1] - r[e0]; step = max(2, int(round(side * rel_step))); n = int(span * side / step)
        offs = [k * step for k in range(-n, n + 1)]; cands = []
        for a in offs:
            for b in offs:
                c = list(r); c[e0] += a; c[e1] += b
                if valid(tuple(c), min_side=8):
                    cands.append(tuple(c))
        s, a = sc.score(img, cands, sm, bs=64); r = list(cands[int(np.argmax(s))])
    return tuple(r)


def random_search(sc, img, sm, starts, n=256, span=0.35, top=2, seed=0):
    """Random perturbation of the four edges around the given starts (for U-Net boxes wrong on several edges)."""
    rng = np.random.default_rng(seed); cands = []
    for r in starts:
        h, w = r[1] - r[0], r[3] - r[2]
        for _ in range(n):
            d = rng.uniform(-span, span, 4) * np.array([h, h, w, w])
            q = tuple(int(round(v)) for v in (max(0, r[0] + d[0]), min(512, r[1] + d[1]), max(0, r[2] + d[2]), min(512, r[3] + d[3])))
            if valid(q, min_side=8, max_aspect=16.0):
                cands.append(q)
    s, a = sc.score(img, cands, sm, bs=64); order = np.argsort(-s); out = []
    for k in order:
        if all(rect_iou(cands[k], q) < 0.6 for q in out):
            out.append(cands[k])
        if len(out) >= top:
            break
    best = None
    for r1 in out:
        r2 = edge_scan(sc, img, sm, r1)
        rr, sf, acc = refine(sc, img, sm, r2, max_aspect=16.0, first_step=max(1, int(round(min(r2[1] - r2[0], r2[3] - r2[2]) * 0.04))))
        if best is None or sf > best[1]:
            best = (rr, sf, acc)
        if acc >= 0.95:
            break
    return best


def search(sc, img, sm, p, topk=2, rand=0):
    H = hypotheses(p)
    if not H:
        return {}
    s, a = sc.score(img, H, sm, bs=64); order = np.argsort(-s)
    starts = []
    for k in order:
        if all(rect_iou(H[k], q) < 0.7 for q in starts):
            starts.append(H[k])
        if len(starts) >= topk:
            break
    best = None
    for r0 in starts:
        J = jitter(r0); s, a = sc.score(img, J, sm, bs=64); r1 = J[int(np.argmax(s))]
        r2 = edge_scan(sc, img, sm, r1)
        rr, sf, acc = refine(sc, img, sm, r2, max_aspect=16.0, first_step=max(1, int(round(min(r2[1] - r2[0], r2[3] - r2[2]) * 0.04))))
        if best is None or sf > best[1]:
            best = (rr, sf, acc, r0)
        if acc >= 0.95:
            break
    out = dict(n_hyp=len(H), rect_init=[int(v) for v in best[3]])
    if 0.62 <= best[2] < 0.9:                       # ambiguous: partly located? try a joint scan of opposite edges
        r3 = pair_scan(sc, img, sm, best[0])
        rr, sf, acc = refine(sc, img, sm, r3, max_aspect=16.0, first_step=max(1, int(round(min(r3[1] - r3[0], r3[3] - r3[2]) * 0.04))))
        out["pair"] = float(acc)
        if sf > best[1]:
            best = (rr, sf, acc, best[3])
    if rand and 0.62 <= best[2] < 0.9:
        rb = random_search(sc, img, sm, [tuple(best[0])] + [q for q in starts if rect_iou(q, best[0]) < 0.7][:1], n=rand)
        out["rand"] = float(rb[2])
        if rb[1] > best[1]:
            best = (rb[0], rb[1], rb[2], best[3])
    out.update(rect_hc=[int(v) for v in best[0]], soft_hc=float(best[1]), acc_hc=float(best[2]))
    if best[2] >= 0.62:
        for tag, pl in (("pm", False), ("pp", True)):
            rp = polish(sc, img, sm, best[0], pl); sp, ap = sc.score(img, [rp], sm)
            out["rect_" + tag] = [int(v) for v in rp]; out["acc_" + tag] = float(ap[0]); out["soft_" + tag] = float(sp[0])
    out.update(rect=out["rect_hc"], soft=out["soft_hc"], acc=out["acc_hc"])
    return out


def capped_session(wm, gb):
    """Rebuild the ONNX session with a GPU memory cap: the default CUDA arena (power-of-two growth, exhaustive cuDNN
    search) was measured at 25 GB for this model, far above our 12 GB budget."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    prov = [("CUDAExecutionProvider", {"device_id": 0, "gpu_mem_limit": int(gb * 2 ** 30), "arena_extend_strategy": "kSameAsRequested",
                                       "cudnn_conv_algo_search": "HEURISTIC"})]
    path = f"{OUT}/stega_dec_only.onnx"      # decoder-only sub-graph (ss_make_dec.py): identical logits, no encoder pass
    wm.sess = None
    wm.sess = ort.InferenceSession(path, so, providers=prov)
    wm._run = lambda x, outputs, secret=None: wm.sess.run(outputs, {"image": x.astype(np.float32)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decoy", type=int, default=0); ap.add_argument("--names", default=""); ap.add_argument("--shard", default="0/1"); ap.add_argument("--mem_gb", type=float, default=4.0); ap.add_argument("--rand", type=int, default=0)
    ap.add_argument("--prev", default="", help="raw results of a previous run: only images with 0.62 <= acc < 0.9 are searched again")
    a = ap.parse_args()
    wm = load_wm("stegastamp"); capped_session(wm, a.mem_gb); sc = Scorer(wm)
    z = np.load(f"{OUT}/whole_{a.split}.npz"); whole = {str(n): float(v) for n, v in zip(z["names"], z["acc"])}
    z = np.load(f"{W}/wm/out/stegastamp_ftA_{a.split}probs.npz"); P = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
    items = load_items(a.split, "stegastamp")
    if a.names:
        items = [it for it in items if it["name"] in a.names.split(",")]
    if a.limit:
        items = items[:a.limit]
    k0, kn = [int(v) for v in a.shard.split("/")]
    items = [it for it in items if whole[it["name"]] < 0.8][k0::kn] + ([it for it in items if whole[it["name"]] >= 0.8] if k0 == 0 else [])
    G = gt128() if a.split == "train" else {}
    res, t0 = {}, time.time(); prev = {}
    for f in [f for f in a.prev.split(",") if f]:
        prev.update(json.load(open(f"{OUT}/{f}")))
    for n, it in enumerate(items):
        r = dict(acc_whole=whole[it["name"]])
        if it["name"] in prev and not (0.62 <= prev[it["name"]].get("acc", 0) < 0.9):
            res[it["name"]] = prev[it["name"]]; continue
        if whole[it["name"]] < 0.8 and it["name"] in P:
            img = load_img(it["path"]); sm = sm_of(it["bits"])
            if a.decoy:
                gg = torch.Generator(device="cuda").manual_seed(a.decoy * 100003 + n)
                sm = (torch.rand(100, device="cuda", generator=gg) > 0.5).float() * 2 - 1
            t1 = time.time(); n0 = sc.n_eval; r.update(search(sc, img, sm, P[it["name"]], rand=a.rand)); r["t"] = time.time() - t1; r["n_eval"] = sc.n_eval - n0
        if it["name"] in G:
            gt = G[it["name"]]; r["kind"] = kind(gt); r["full"] = bool(it["full"])
            m5 = gt512(it["name"]); ys, xs = np.nonzero(m5)
            if len(ys):
                r["gt_bbox"] = [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]
                for tag in ("hc", "pm", "pp"):
                    if "rect_" + tag in r:
                        r["iou_" + tag] = rect_iou(r["rect_" + tag], r["gt_bbox"])
        res[it["name"]] = r
        print(n, len(items), f"{time.time() - t0:.0f}s", it["name"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if n % 10 == 0:
            json.dump(res, open(f"{OUT}/{a.out}", "w"))
    json.dump(res, open(f"{OUT}/{a.out}", "w"))
    print("DONE", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
