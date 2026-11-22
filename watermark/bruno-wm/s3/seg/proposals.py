"""Segment proposals for every image. Usage: ./sg 1 seg/proposals.py <train|test> [n cpu workers] [limit]

Per image, at R = 256 (image downscaled 2x, INTER_AREA), index 0 = coarsest scale:
  slic  3 scales  SLIC superpixels (torch, GPU), 8x8 / 16x16 / 32x32 seeds, connectivity enforced
  fh    3 scales  Felzenszwalb-Huttenlocher merging run on the minimum spanning tree of the 8-connected Lab graph
  ws    2 scales  marker watershed (cv2.watershed), markers = regional minima of the smoothed Lab gradient
  ms    2 scales  mean-shift filtering (cv2.pyrMeanShiftFiltering) + connected components of similar colours
  inst            torchvision Mask R-CNN (resnet50 fpn v2, COCO), score > 0.02, up to 100 masks, near-duplicates removed
Output: <OUT>/seg/<split>/<name>.npz (see seglib.load_props).
"""
import os
import sys
import time

import cv2
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from common import OUT, load_items

R = 256
N = R * R
cv2.setNumThreads(1)
_I = np.arange(N).reshape(R, R)
PA = np.concatenate([_I[:, :-1].ravel(), _I[:-1].ravel()])  # 4-neighbour pixel pairs
PB = np.concatenate([_I[:, 1:].ravel(), _I[1:].ravel()])
EA = np.concatenate([PA, _I[:-1, :-1].ravel(), _I[:-1, 1:].ravel()])  # 8-neighbour pairs
EB = np.concatenate([PB, _I[1:, 1:].ravel(), _I[1:, :-1].ravel()])


def cc_pairs(same):
    """Connected components of the pixel graph keeping only the 4-neighbour pairs flagged in `same`."""
    g = sp.coo_matrix((np.ones(int(same.sum()), np.uint8), (PA[same], PB[same])), shape=(N, N))
    return connected_components(g, directed=False)[1]


def merge_small(f, feat, min_size):
    """Merge every segment smaller than min_size into its most similar (mean Lab colour) neighbour."""
    f = np.unique(f.ravel(), return_inverse=True)[1]
    feat = feat.reshape(N, 3)
    for _ in range(10):
        n = int(f.max()) + 1
        size = np.bincount(f, minlength=n)
        small = size < min_size
        if not small.any():
            break
        a, b = f[PA], f[PB]
        m = a != b
        a, b = a[m].astype(np.int64), b[m].astype(np.int64)
        e = np.unique(np.concatenate([a * n + b, b * n + a]))
        ea, eb = e // n, e % n
        k = small[ea]
        ea, eb = ea[k], eb[k]
        mean = np.stack([np.bincount(f, weights=feat[:, c], minlength=n) for c in range(3)], 1) / size[:, None]
        d = np.linalg.norm(mean[ea] - mean[eb], axis=1)
        o = np.lexsort((d, ea))
        ea, eb = ea[o], eb[o]
        first = np.r_[True, ea[1:] != ea[:-1]]
        g = sp.coo_matrix((np.ones(int(first.sum()), np.uint8), (ea[first], eb[first])), shape=(n, n))
        f = np.unique(connected_components(g, directed=False)[1][f], return_inverse=True)[1]
    return f.reshape(R, R)


# ---------------------------------------------------------------- SLIC (GPU, no host sync inside)
SLIC_N = (8, 16, 32)


@torch.no_grad()
def slic_raw(lab_t, n, m=8.0, iters=6):
    """lab_t: 3 x R x R Lab tensor on the GPU, n: seeds per side (R % n == 0). Returns R x R long labels (raw:
    segments may be disconnected, slic_post fixes that on the CPU)."""
    dev = lab_t.device
    S = R // n
    yy, xx = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing="ij")
    feat = torch.cat([lab_t, yy[None].float() * m / S, xx[None].float() * m / S])
    cen = F.avg_pool2d(feat[None], S)[0].reshape(5, -1)
    by, bx = yy // S, xx // S
    oks, ids = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cy, cx = by + dy, bx + dx
            oks.append((cy >= 0) & (cy < n) & (cx >= 0) & (cx < n))
            ids.append(cy.clamp(0, n - 1) * n + cx.clamp(0, n - 1))
    oks, ids = torch.stack(oks), torch.stack(ids)  # 9 x R x R
    inf = torch.tensor(float("inf"), device=dev)
    one = torch.ones(R * R, device=dev)
    for _ in range(iters):
        d = ((feat[:, None] - cen[:, ids]) ** 2).sum(0)  # 9 x R x R
        lab = ids.gather(0, torch.where(oks, d, inf).argmin(0)[None])[0]
        fl = lab.reshape(-1)
        cnt = torch.zeros(n * n, device=dev).index_add_(0, fl, one)
        new = torch.zeros(5, n * n, device=dev).index_add_(1, fl, feat.reshape(5, -1)) / cnt.clamp(min=1)
        cen = torch.where(cnt[None] > 0, new, cen)
    return lab


def slic_post(raw, labimg):
    out = []
    for n, f in zip(SLIC_N, raw):
        f = f.ravel()
        out.append(merge_small(cc_pairs(f[PA] == f[PB]), labimg, max(4, (R // n) ** 2 // 4)))
    return out


# ---------------------------------------------------------------- Felzenszwalb on the MST
def fh_all(labimg, ks=(400.0, 120.0, 40.0), mins=(150, 60, 25)):
    f = labimg.reshape(N, 3)
    w = np.linalg.norm(f[EA] - f[EB], axis=1).astype(np.float64) + 1e-3
    t = minimum_spanning_tree(sp.coo_matrix((w, (EA, EB)), shape=(N, N)).tocsr()).tocoo()
    o = np.argsort(t.data, kind="stable")
    us, vs, ws = t.row[o].tolist(), t.col[o].tolist(), t.data[o].tolist()
    out = []
    for k, mn in zip(ks, mins):
        par = list(range(N)); size = [1] * N; thr = [k] * N
        for a, b, x in zip(us, vs, ws):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            while par[b] != b:
                par[b] = par[par[b]]; b = par[b]
            if a != b and x <= thr[a] and x <= thr[b]:
                if size[a] < size[b]:
                    a, b = b, a
                par[b] = a; size[a] += size[b]; thr[a] = x + k / size[a]
        p = np.array(par)
        while True:
            q = p[p]
            if (q == p).all():
                break
            p = q
        out.append(merge_small(p, labimg, mn))
    return out


# ---------------------------------------------------------------- watershed
def ws_all(bgr8, labimg, radii=(9, 4), mins=(60, 20)):
    sm = cv2.GaussianBlur(labimg, (0, 0), 1.0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0); gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1)
    grad = np.sqrt((gx ** 2 + gy ** 2).sum(2))
    img = cv2.GaussianBlur(bgr8, (0, 0), 1.0)
    out = []
    for r, mn in zip(radii, mins):
        g = cv2.GaussianBlur(grad, (0, 0), r / 3.0)
        mk = cv2.connectedComponents((g <= cv2.erode(g, np.ones((2 * r + 1, 2 * r + 1), np.uint8))).astype(np.uint8))[1]
        mk = cv2.watershed(img, mk.astype(np.int32))
        bad = mk <= 0
        if bad.any():
            iy, ix = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
            mk = mk[iy, ix]
        f = mk.ravel()
        out.append(merge_small(cc_pairs(f[PA] == f[PB]), labimg, mn))
    return out


# ---------------------------------------------------------------- mean shift
def ms_all(bgr8, labimg, sp_=8, sr=20, tols=(10.0, 4.0), mins=(100, 30)):
    f = cv2.pyrMeanShiftFiltering(bgr8, sp_, sr, maxLevel=1).reshape(N, 3).astype(np.int16)
    dif = np.abs(f[PA] - f[PB]).max(1)
    return [merge_small(cc_pairs(dif <= tol), labimg, mn) for tol, mn in zip(tols, mins)]


# ---------------------------------------------------------------- Mask R-CNN
def load_maskrcnn():
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights as Wt, maskrcnn_resnet50_fpn_v2
    net = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, box_score_thresh=0.02,
                                   box_detections_per_img=100, rpn_post_nms_top_n_test=1000)
    p = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", os.path.basename(Wt.COCO_V1.url))
    net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    return net.eval().cuda()


@torch.no_grad()
def inst_batch(net, imgs):
    """imgs: list of 3 x 512 x 512 tensors in [0,1]. Returns per image (masks N x R x R bool, score, label, box),
    sorted by decreasing score, near-duplicates (IoU > 0.9 with a better-scoring mask) and tiny masks removed."""
    res = []
    for o in net(imgs):
        m = F.avg_pool2d(o["masks"], 2)[:, 0] > 0.5
        fl = m.reshape(len(m), R * R).float()  # (also valid when there is no detection at all)
        area = fl.sum(1)
        inter = fl @ fl.T
        iou = (inter / (area[:, None] + area[None] - inter).clamp(min=1)).cpu().numpy()
        area = area.cpu().numpy()
        keep = []
        for i in range(len(area)):
            if area[i] >= 16 and not (keep and iou[i, keep].max() > 0.9):
                keep.append(i)
        keep = np.array(keep, dtype=np.int64)
        res.append((m.cpu().numpy()[keep], o["scores"].cpu().numpy()[keep].astype(np.float32),
                    o["labels"].cpu().numpy()[keep].astype(np.int16), o["boxes"].cpu().numpy()[keep].astype(np.float32)))
    return res


def prep(path):
    rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert rgb.shape == (512, 512, 3), rgb.shape
    small = cv2.resize(rgb, (R, R), interpolation=cv2.INTER_AREA)
    labimg = cv2.cvtColor(small.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    return rgb, small, labimg, cv2.GaussianBlur(labimg, (0, 0), 0.8)


def cpu_part(task):
    """Runs in a worker process: classical segmentations + SLIC clean-up + save."""
    dst, path, raw, masks, sc, lb, bx = task
    T = {}
    rgb, small, labimg, labsm = prep(path)
    bgr8 = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    res = {}
    t = time.time(); res["slic"] = slic_post(raw, labimg); T["slic"] = time.time() - t
    t = time.time(); res["fh"] = fh_all(labsm); T["fh"] = time.time() - t
    t = time.time(); res["ws"] = ws_all(bgr8, labimg); T["ws"] = time.time() - t
    t = time.time(); res["ms"] = ms_all(bgr8, labimg); T["ms"] = time.time() - t
    t = time.time()
    nseg = {k: [int(x.max()) + 1 for x in v] for k, v in res.items()}
    nseg["inst"] = [len(sc)]
    res = {k: np.stack(v).astype(np.uint16) for k, v in res.items()}
    np.savez_compressed(dst + ".tmp.npz", inst_bits=np.packbits(masks.reshape(len(masks), -1), axis=1) if len(masks)
                        else np.zeros((0, N // 8), np.uint8), inst_score=sc, inst_label=lb, inst_box=bx, **res)
    os.replace(dst + ".tmp.npz", dst); T["save"] = time.time() - t
    return T, nseg


def main():
    import multiprocessing as mp
    split = sys.argv[1]
    ncpu = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    # `sg` pins every step to ONE core (cpu 48) shared by all sessions. For the CPU-bound classical methods, use
    # `ncpu` otherwise idle cores of our own (fully allocated) node instead of saturating that shared core.
    orig = os.sched_getaffinity(0)
    if ncpu > 1:
        try:
            os.sched_setaffinity(0, range(os.cpu_count()))
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-ncpu:])
        except OSError as e:
            print("cannot widen cpu affinity:", e); os.sched_setaffinity(0, orig)
    ncpu = min(ncpu, len(os.sched_getaffinity(0)))
    print("cores", sorted(os.sched_getaffinity(0)), "workers", ncpu, "load", os.getloadavg(), flush=True)
    items = load_items(split)[:limit]
    os.makedirs(f"{OUT}/seg/{split}", exist_ok=True)
    todo = [it for it in items if limit is not None or not os.path.exists(f"{OUT}/seg/{split}/{it['name']}.npz")]
    pool = mp.get_context("fork").Pool(ncpu)  # forked before CUDA is initialised
    net = load_maskrcnn()
    tg = [0.0, 0.0]
    t00 = time.time()

    def gen(bs=8):
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            pre = [prep(it["path"]) for it in chunk]
            t = time.time()
            ins = inst_batch(net, [torch.from_numpy(p[0]).cuda().permute(2, 0, 1).float() / 255 for p in pre])
            tg[0] += time.time() - t; t = time.time()
            raws = []
            for p in pre:
                lt = torch.from_numpy(p[3]).cuda().permute(2, 0, 1)
                raws.append(torch.stack([slic_raw(lt, n) for n in SLIC_N]).to(torch.int16))
            raws = torch.stack(raws).cpu().numpy()
            tg[1] += time.time() - t
            for it, raw, (masks, sc, lb, bx) in zip(chunk, raws, ins):
                yield (f"{OUT}/seg/{split}/{it['name']}.npz", it["path"], raw, masks, sc, lb, bx)

    T, NS, n = {}, {}, 0
    for Ti, ns in pool.imap_unordered(cpu_part, gen()):
        n += 1
        for k, v in Ti.items():
            T[k] = T.get(k, 0.0) + v
        for k, v in ns.items():
            NS.setdefault(k, []).append(v)
        if n % 100 == 0 or n == len(todo):
            print(f"{split} {n}/{len(todo)} wall {time.time() - t00:.0f}s | gpu s/img: maskrcnn {tg[0] / n:.3f} slic {tg[1] / n:.3f}"
                  " | cpu s/img (1 core): " + " ".join(f"{k} {v / n:.3f}" for k, v in T.items())
                  + " | mean #seg: " + " ".join(f"{k} {np.mean(v, 0).round(0).tolist()}" for k, v in NS.items()), flush=True)
    pool.close(); pool.join()
    print("done", split, f"{time.time() - t00:.0f}s", f"gpu {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")


if __name__ == "__main__":
    main()
