"""Production recipe: sliding-patch occlusion evidence map + decoder-verified hole size.
usage: final.py <method> <train|test> <stat mean|tstat> <smooth> <cfg patch:stride:fill> [more cfgs, maps are averaged]
  train: the 85 images with full=True, prints AUC / cross-validated mIoU, saves OUT/occl_<method>_train.npz
  test : the test images whose whole-image bit accuracy is > 0.70 (OUT/whole_test.npz), saves OUT/occl_<method>_test.npz
Saved arrays: names, maps (float16, N x 128 x 128, higher = watermarked, each map divided by its median),
m0 (whole-image margin), acc_whole, qs and ratio (N x len(qs)): margin after filling the lowest-q fraction of the
map with the mean colour, divided by m0 (the VERIFY step: a true hole can be filled without loss of margin).
Rules that do not use the ground truth:
  t-rule     : hole = map < t                      (map is in units of its median)
  verify rule: hole = lowest q* fraction, q* = largest q such that ratio(q') >= 1 - eps for every q' <= q"""
import sys, time
from lib import *

meth, split, stat, smooth = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
cfgs = sys.argv[5:]
NIMG = int(os.environ.get("NIMG", 0)); TAG = os.environ.get("TAG", "")
set_log(f"final_{meth}_{split}{TAG}"); wm = load_wm(meth)
if split == "train":
    items = [it for it in load_items("train", meth) if it["full"]]
else:
    z = np.load(f"{OUT}/whole_test.npz")
    ok = {str(n) for n, a, m in zip(z["names"], z["acc"], z["method"]) if a > 0.70 and str(m) == meth}
    items = [it for it in load_items("test", meth) if it["name"] in ok]
if NIMG:
    items = items[:NIMG]
names = [it["name"] for it in items]
part = [it for it in load_items("train", meth) if not it["full"]]   # mostly clean images, source of the "other" fill
VQ = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]

plans = []
for cfg in cfgs:
    s, st, fk = cfg.split(":"); s = int(s); st = int(st)
    pos = [(y, x) for y in range(0, 512 - s + 1, st) for x in range(0, 512 - s + 1, st)]
    cov = torch.zeros(len(pos), 128, 128, device=DEV)
    for k, (y, x) in enumerate(pos):
        cov[k, y // 4:(y + s) // 4, x // 4:(x + s) // 4] = 1
    plans.append((s, fk, pos, cov, cov.sum(0).clamp(min=1)))
nq = sum(len(p[2]) for p in plans) + len(VQ) + 1
maps, ratios, m0s, accs = [], [], [], []
t0 = time.time()
for j, it in enumerate(items):
    img = load_img(it["path"]); sm = sm_of(it["bits"])
    oth = load_img(part[j % len(part)]["path"])
    y0 = slog(wm, img, sm); m0 = y0.mean().item()
    ev = torch.zeros(128, 128, device=DEV)
    for s, fk, pos, cov, cnt in plans:
        fill = make_fill(img, fk, oth)
        D = []
        for i in range(0, len(pos), 128):
            xb = img.repeat(len(pos[i:i + 128]), 1, 1, 1)
            for b, (y, x) in enumerate(pos[i:i + 128]):
                xb[b, :, y:y + s, x:x + s] = fill[0, :, y:y + s, x:x + s]
            D.append(y0 - slog(wm, xb, sm))
        D = torch.cat(D)
        d = D.mean(1) if stat == "mean" else D.mean(1) / (D.std(1) + 1e-6)
        e = torch.einsum("k,khw->hw", d, cov) / cnt
        ev += e / e.median().abs().clamp(min=1e-6) / len(plans)
    if smooth > 0:
        ev = gauss_blur(ev[None, None], smooth)[0, 0]
    ev = ev / ev.median().abs().clamp(min=1e-6)
    # verify step: fill the lowest-q fraction with the mean colour and read the margin again
    srt = ev.flatten().sort().values; fill = make_fill(img, "mean")
    hm = torch.stack([(ev < srt[int(q * (srt.numel() - 1))]).float() for q in VQ])[:, None]      # 1 = candidate hole
    hm = F.interpolate(hm, size=(512, 512), mode="nearest")
    r = slog(wm, hm * fill + (1 - hm) * img, sm).mean(1) / max(m0, 1e-6)
    maps.append(ev.cpu().numpy()); ratios.append(r.cpu().numpy()); m0s.append(m0); accs.append((y0 > 0).float().mean().item())
dt = (time.time() - t0) / max(len(items), 1)
maps = np.array(maps, np.float32); ratios = np.array(ratios, np.float32)
print(f"== final {meth} {split} stat={stat} smooth={smooth} cfgs={cfgs}: {len(items)} images, {nq} queries, {dt:.2f} s/image", flush=True)
np.savez(f"{OUT}/occl_{meth}_{split}{TAG}.npz", names=np.array(names), maps=maps.astype(np.float16), m0=np.array(m0s, np.float32),
         acc_whole=np.array(accs, np.float32), qs=np.array(VQ, np.float32), ratio=ratios)

if split == "train":
    G = gt128()
    evaluate(maps, names, G, f"final {meth}")
    # verify rule, eps cross-validated
    EPS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
    N = len(names); S = np.zeros((N, len(EPS))); QC = np.zeros((N, len(EPS)))
    for i in range(N):
        srt = np.sort(maps[i].ravel())
        for j, e in enumerate(EPS):
            okq = np.cumprod(ratios[i] >= 1 - e).sum()            # number of leading q's that pass
            q = VQ[okq - 1] if okq > 0 else 0.0
            pred = maps[i] >= srt[int(q * (srt.size - 1))] if q > 0 else np.ones((128, 128), bool)
            S[i, j] = miou(pred, G[names[i]]); QC[i, j] = q
    rng = np.random.RandomState(0); perm = rng.permutation(N); fold = np.empty(N, int); fold[perm] = np.arange(N) % 5
    cv = np.zeros(N); ch = []
    for f in range(5):
        j = int(np.argmax(S[fold != f].mean(0))); cv[fold == f] = S[fold == f, j]; ch.append(EPS[j])
    hole = np.array([1 - G[n].mean() for n in names])
    b = " ".join(f"{lab}:{cv[(hole >= a) & (hole < bb)].mean():.3f}" for a, bb, lab in BINS)
    print(f"verify rule: cv mIoU {cv.mean():.3f} (eps chosen {ch}) | per eps " + " ".join(f"{e}:{s:.3f}" for e, s in zip(EPS, S.mean(0)))
          + f" | {b}")
    jb = int(np.argmax(S.mean(0)))
    print(f"   predicted hole fraction vs true (eps={EPS[jb]}): corr {np.corrcoef(QC[:, jb], hole)[0, 1]:.2f}, "
          f"mean pred {QC[:, jb].mean():.3f} true {hole.mean():.3f}")
