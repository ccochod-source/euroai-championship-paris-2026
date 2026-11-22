"""Does session 75's template-informed network (trained on kit composites only) work on REAL full images once the
template is computed on the estimated canvas instead of the visible frame?
usage: corr_eval.py <method> <ckpt tag e.g. i1_last> [split]"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from common import *
T75 = "/p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/bruno/informed"
sys.path.append(T75)
from corrnet import CorrNet  # noqa: E402
from templates import Templater, normalise  # noqa: E402
from scipy.stats import rankdata
meth, tag = sys.argv[1], sys.argv[2]; split = sys.argv[3] if len(sys.argv) > 3 else "train"
wm = load_wm(meth); templ = Templater(wm); dev = "cuda"
net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(f"{T75}/out/{meth}_{tag}.pt", map_location="cpu", weights_only=True))
import glob
cv = {}
for f in sorted(glob.glob(f"{OUT}/frame_{meth}_{split}*.json")): cv.update(json.load(open(f)))
g = gt128() if split == "train" else {}

def canvas(x, a, b, l=0, r=0):
    def idx(lo, hi):
        i = np.arange(-lo, 512 + hi); p = 2 * 511; i = np.mod(i, p); return torch.from_numpy(np.where(i > 511, p - i, i)).to(x.device)
    return x[:, :, idx(a, b), :][:, :, :, idx(l, r)]
def auc(score, pos):
    r = rankdata(score); n1 = pos.sum(); n0 = (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

@torch.no_grad()
def prob(x255, t):
    with torch.autocast("cuda", dtype=torch.float16):
        out, _ = net(x255, t)
    return torch.sigmoid(out.float())[0].cpu().numpy()

names, P0, P1, meta = [], [], [], []
for it in load_items(split, meth):
    if it["name"] not in cv: continue
    c = cv[it["name"]]; x = load_img(it["path"]) * 255; bits = torch.from_numpy(it["bits"].astype(np.float32)).to(dev)[None]
    t0, _ = normalise(templ(x, bits))
    a, b, l, r = c["a"], c["b"], c.get("l", 0), c.get("r", 0); t1, _ = normalise(templ(canvas(x, a, b, l, r), bits)[:, :, a:a + 512, l:l + 512])
    names.append(it["name"]); P0.append(prob(x, t0)); P1.append(prob(x, t1)); meta.append((c["slope"], c["slope_plain"], a + b))
P0, P1, M = np.array(P0), np.array(P1), np.array(meta)
np.savez_compressed(f"{OUT}/corrsq_{meth}_{split}_{tag}.npz", names=np.array(names), probs=P1.astype(np.float16), probs_plain=P0.astype(np.float16), slope=M[:, 0], slope_plain=M[:, 1])
if split == "train":
    G = np.array([g[n] for n in names]); ones = np.mean([miou(np.ones((128, 128)), x) for x in G])
    print(f"== {meth} {tag}: n={len(names)} real full images, all-ones {ones:.3f}")
    for lab, P in [("template on the visible frame ", P0), ("template on the estimated canvas", P1)]:
        au = np.nanmean([auc(p.ravel(), x.ravel()) for p, x in zip(P, G)])
        print(f"   {lab}: AUC {au:.3f} | mean prob on watermarked cells {np.mean([p[x].mean() for p, x in zip(P, G)]):.3f}, on hole cells {np.nanmean([p[~x].mean() for p, x in zip(P, G) if (~x).any()]):.3f} | mIoU " + " ".join(f"t{t}: {np.mean([miou(p > t, x) for p, x in zip(P, G)]):.3f}" for t in [0.1, 0.3, 0.5, 0.7]))
    for t in [0.8, 0.9, 0.95]: print(f"      canvas-template mIoU at t{t}: {np.mean([miou(p > t, x) for p, x in zip(P1, G)]):.3f}")
    hole = 1 - G.reshape(len(G), -1).mean(1); sc = np.array([miou(p > 0.5, x) for p, x in zip(P1, G)])
    print("      by hole size (<2%, 2-6%, 6-12%, >12%): " + " ".join(f"{sc[(hole >= lo) & (hole < hi)].mean():.3f}" for lo, hi in [(0, .02), (.02, .06), (.06, .12), (.12, 1)]))
    ok = M[:, 0] > 0.2
    for lab, s in [("canvas found (slope > 0.2)", ok), ("canvas not found", ~ok)]:
        if s.sum(): print(f"   {lab}: n={s.sum()} mIoU canvas-template t0.5 {np.mean([miou(p > 0.5, x) for p, x in zip(P1[s], G[s])]):.3f} | visible-frame template {np.mean([miou(p > 0.5, x) for p, x in zip(P0[s], G[s])]):.3f} | all-ones {np.mean([miou(np.ones((128,128)), x) for x in G[s]]):.3f}")
