"""Segment proposals: loading, oracle masks and `snap` (evidence map -> segment-aligned mask).

Import from anywhere in s3 with `from seg.seglib import load_props, partitions, snap` (s3/ is on PYTHONPATH).
All label maps are int arrays R x R with R = 256 (1 label-map pixel = 2 x 2 image pixels; 1 evaluation cell = 2 x 2
label-map pixels). Labels are 0..n-1, every segment of a partition is 4-connected (except `inst` cells and grids).
"""
import os

import numpy as np

R = 256
SEG_DIR = os.environ.get("S3_SEG", "/p/scratch/training2666/dasilvalopes3/wm/s3/out/seg")
# name of every partition available through partitions(): source + scale index (0 = coarsest)
CLASSICAL = ["slic0", "slic1", "slic2", "fh0", "fh1", "fh2", "ws0", "ws1", "ms0", "ms1"]


def load_props(name, split="train"):
    """dict: slic (3,R,R), fh (3,R,R), ws (2,R,R), ms (2,R,R) uint16 label maps (index 0 = coarsest scale);
    inst (N,R,R) bool Mask R-CNN instance masks sorted by decreasing score; inst_score (N,), inst_label (N,) COCO ids,
    inst_box (N,4) xyxy in 512-pixel coordinates."""
    z = np.load(f"{SEG_DIR}/{split}/{name}.npz")
    d = {k: z[k] for k in z.files}
    n = len(d["inst_score"])
    d["inst"] = np.unpackbits(d.pop("inst_bits"), axis=1)[:, :R * R].reshape(n, R, R).astype(bool)
    return d


def inst_cells(inst, top=60):
    """Partition induced by the instance masks: two pixels are in the same cell when they are covered by exactly the
    same set of (top-scoring) instances. Cell 0 = background (covered by no instance) when it exists."""
    code = np.zeros((R, R), np.uint64)
    for i, m in enumerate(inst[:top]):
        code |= m.astype(np.uint64) << np.uint64(i)
    u, lab = np.unique(code, return_inverse=True)
    return lab.reshape(R, R)


def refine(a, b):
    """Common refinement of two partitions."""
    return np.unique(a.astype(np.int64) * (int(b.max()) + 1) + b, return_inverse=True)[1].reshape(a.shape)


def grid(n):
    """Blind control: n x n squares."""
    i = np.arange(R) * n // R
    return i[:, None] * n + i[None]


def partitions(d):
    """dict name -> label map for every stored partition, plus `inst` (cells of the Mask R-CNN masks)."""
    p = {f"{s}{i}": d[s][i].astype(np.int64) for s in ("slic", "fh", "ws", "ms") for i in range(len(d[s]))}
    p["inst"] = inst_cells(d["inst"])
    return p


# ---------------------------------------------------------------- oracle (needs the ground truth)
def up2(a):
    return a.repeat(2, 0).repeat(2, 1)


def oracle_partition(lab, gt512, box=False):
    """Best 512 x 512 mask made of whole segments: a segment is kept when more than half of its pixels are inside
    the ground truth. box=True: segments are first clipped to the bounding rectangle of the ground truth."""
    lab = up2(lab); n = int(lab.max()) + 1
    if box:
        ys, xs = np.nonzero(gt512)
        if len(ys) == 0:
            return np.zeros_like(gt512)
        bm = np.zeros_like(gt512); bm[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        tot = np.bincount(lab[bm], minlength=n); ins = np.bincount(lab[bm & gt512], minlength=n)
        return (ins * 2 > tot)[lab] & bm
    tot = np.bincount(lab.ravel(), minlength=n); ins = np.bincount(lab[gt512], minlength=n)
    return (ins * 2 > tot)[lab]


def oracle_inst_union(inst, gt512):
    """Union of the instance masks that are more than half inside the ground truth (no background segment)."""
    out = np.zeros_like(gt512)
    for m in inst:
        m = up2(m)
        if (m & gt512).sum() * 2 > m.sum():
            out |= m
    return out


# ---------------------------------------------------------------- snap (no ground truth)
def _ev256(ev128):
    import cv2
    return cv2.resize(np.asarray(ev128, np.float32), (R, R), interpolation=cv2.INTER_LINEAR)


def _down(mask256):
    return mask256.reshape(128, 2, 128, 2).mean((1, 3)) >= 0.5


def seg_mean(ev256, lab):
    n = int(lab.max()) + 1
    cnt = np.bincount(lab.ravel(), minlength=n)
    return np.bincount(lab.ravel(), weights=ev256.ravel(), minlength=n) / np.maximum(cnt, 1), cnt


def snap(evidence128, proposals, thr=0.5, margin=None, min_px=0):
    """Segment-aligned mask from a (noisy, low-resolution) evidence map.

    evidence128: 128 x 128 float array, larger = more likely watermarked (threshold `thr`).
    proposals:   one label map (R x R) or a list of label maps ordered COARSE -> FINE.
    margin=None: hard snap with the LAST label map only: a segment is 1 when its mean evidence is > thr.
    margin=m:    a pixel takes the decision of the coarsest segment that is confident (|mean - thr| >= m and at
                 least min_px pixels); pixels with no confident segment keep the evidence map's own decision.
    Returns a 128 x 128 bool mask."""
    labs = proposals if isinstance(proposals, (list, tuple)) else [proposals]
    ev = _ev256(evidence128)
    if margin is None:
        mean, _ = seg_mean(ev, labs[-1])
        return _down((mean > thr)[labs[-1]])
    out = ev > thr
    done = np.zeros((R, R), bool)
    for lab in labs:
        mean, cnt = seg_mean(ev, lab)
        conf = (np.abs(mean - thr) >= margin) & (cnt >= min_px)
        take = conf[lab] & ~done
        out[take] = (mean > thr)[lab][take]
        done |= take
    return _down(out)


def snap_fit(evidence128, lab, cell=8, thr=0.5, lam=0.02):
    """Deconvolution-aware snap for an evidence map whose true resolution is `cell` x `cell` evaluation cells
    (cell=8 -> a 16 x 16 map): find one value x_s in [0,1] per segment such that the segment mask, averaged on those
    coarse cells, reproduces the evidence (bounded least squares, small ridge towards the segment's mean evidence),
    then threshold x. Unlike snap(), a small segment next to the boundary is not decided by blurred evidence."""
    import scipy.sparse as sp
    from scipy.optimize import lsq_linear
    ev = np.asarray(evidence128, np.float32)
    n = int(lab.max()) + 1; c = 128 // cell; k = R // c
    cid = (np.arange(R) // k)[:, None] * c + (np.arange(R) // k)[None]
    A = sp.coo_matrix((np.full(R * R, 1.0 / k ** 2), (cid.ravel(), lab.ravel())), shape=(c * c, n)).tocsr()
    h = cell // 2
    e = ev.reshape(c, cell, c, cell)[:, h - 1:h + 1, :, h - 1:h + 1].mean((1, 3)).ravel()  # value at the cell centre
    x0, _ = seg_mean(_ev256(ev), lab)
    Aa = sp.vstack([A, np.sqrt(lam) * sp.identity(n)]).tocsr()
    x = lsq_linear(Aa, np.concatenate([e, np.sqrt(lam) * np.clip(x0, 0, 1)]), bounds=(0, 1), max_iter=30).x
    return _down((x > thr)[lab])


# ---------------------------------------------------------------- CPU helper
def idle_pool(ncpu, skip=16):
    """`sg` pins every step to one core shared by all sessions; move this (CPU-only) process to `ncpu` otherwise idle
    cores of our own node and return a fork Pool. Call BEFORE touching CUDA."""
    import multiprocessing as mp
    orig = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, range(os.cpu_count()))
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-(skip + ncpu):-skip])  # the last `skip` cores: proposals.py
    except OSError:
        os.sched_setaffinity(0, orig)
    return mp.get_context("fork").Pool(min(ncpu, len(os.sched_getaffinity(0))))
