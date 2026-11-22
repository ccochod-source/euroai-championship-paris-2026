"""POSITIVE local tests (instead of deletion): show the decoder only a local block and read the margin.
  win:s:st:fill   keep the s x s window, replace everything else by the fill (insertion test)
  tile:s:st       periodic tiling of the s x s block over the whole 512 canvas (block stays at its own position)
  mirror:s:st     same with mirror reflections (continuous at the seams)
usage: tile.py <method> <tag> <cfg> [...]      env NIMG, ROLL=1 (print robustness to large shifts / flips first)
Evidence of a 4x4 cell = mean margin of the blocks covering it."""
import sys, time
from lib import *

meth, tag = sys.argv[1], sys.argv[2]
cfgs = sys.argv[3:]
NIMG = int(os.environ.get("NIMG", 0))
set_log(f"tile_{meth}_{tag}"); wm = load_wm(meth); G = gt128()
items = [it for it in load_items("train", meth) if it["full"]]
if NIMG:
    items = items[:NIMG]
names = [it["name"] for it in items]

if os.environ.get("ROLL"):
    tf = {"image": lambda x: x, "roll16": lambda x: x.roll((16, 16), (2, 3)), "roll64": lambda x: x.roll((64, 64), (2, 3)),
          "roll128": lambda x: x.roll((128, 128), (2, 3)), "roll256": lambda x: x.roll((256, 256), (2, 3)),
          "rollx100": lambda x: x.roll(100, 3), "hflip": lambda x: x.flip(3), "vflip": lambda x: x.flip(2),
          "rot180": lambda x: x.flip(2).flip(3), "transpose": lambda x: x.transpose(2, 3), "rot90": lambda x: x.rot90(1, (2, 3))}
    R = []
    for it in items[:24]:
        img = load_img(it["path"]); sm = sm_of(it["bits"])
        R.append(slog(wm, torch.cat([f(img) for f in tf.values()]), sm).mean(1).cpu().numpy())
    print(f"== {meth}: margin after a global transform (mean over 24 full images): "
          + " ".join(f"{k} {v:+.2f}" for k, v in zip(tf, np.mean(R, 0))))


def idx_maps(kind, pos, s):
    """Source row / column indices (B x 512) of the canvas for each block position."""
    a = torch.arange(512, device=DEV)[None]
    out = []
    for c in (0, 1):
        o = torch.tensor([p[c] for p in pos], device=DEV)[:, None]
        if kind == "tile":
            i = (a - o) % s + o
        else:
            i = (a - o) % (2 * s); i = torch.where(i < s, i, 2 * s - 1 - i) + o
        out.append(i)
    return out


save = dict(names=np.array(names))
for cfg in cfgs:
    f = cfg.split(":"); kind = f[0]; s = int(f[1]); st = int(f[2]); fk = f[3] if len(f) > 3 else "grey"
    pos = [(y, x) for y in range(0, 512 - s + 1, st) for x in range(0, 512 - s + 1, st)]
    cov = torch.zeros(len(pos), 128, 128, device=DEV)
    for k, (y, x) in enumerate(pos):
        cov[k, y // 4:(y + s) // 4, x // 4:(x + s) // 4] = 1
    cnt = cov.sum(0).clamp(min=1)
    ev = {k: [] for k in ["mean", "tstat", "max"]}
    t0 = time.time()
    for it in items:
        img = load_img(it["path"]); sm = sm_of(it["bits"])
        Y = []
        for i in range(0, len(pos), 128):
            pb = pos[i:i + 128]
            if kind == "win":
                fill = make_fill(img, fk)
                xb = fill.repeat(len(pb), 1, 1, 1)
                for b, (y, x) in enumerate(pb):
                    xb[b, :, y:y + s, x:x + s] = img[0, :, y:y + s, x:x + s]
            else:
                iy, ix = idx_maps(kind, pb, s)
                xb = img[0][:, iy[:, :, None], ix[:, None, :]].permute(1, 0, 2, 3)
            Y.append(slog(wm, xb, sm))
        Y = torch.cat(Y)                                           # npos x nbits
        m = Y.mean(1); t = m / (Y.std(1) + 1e-6)
        ev["mean"].append((torch.einsum("k,khw->hw", m, cov) / cnt).cpu().numpy())
        ev["tstat"].append((torch.einsum("k,khw->hw", t, cov) / cnt).cpu().numpy())
        ev["max"].append((m[:, None, None] * cov + (cov - 1) * 1e3).amax(0).cpu().numpy())
    dt = (time.time() - t0) / len(items)
    print(f"== {meth} {cfg}: {len(pos)} queries, {dt:.2f} s/image", flush=True)
    for k, v in ev.items():
        v = np.array(v); save[f"{cfg}|{k}"] = v.astype(np.float32)
        for sm_ in ([0.0, 2.0] if k == "mean" else [0.0]):
            evaluate(to128(v, smooth=sm_), names, G, f"{cfg} {k} sm{sm_}")
np.savez(f"{OUT}/occl_raw_{meth}_{tag}.npz", **save)
