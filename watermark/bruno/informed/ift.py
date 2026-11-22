"""Fine-tune the informed detector on REAL fully watermarked train images, 5-fold cross-validation.
The template of each image is computed on its estimated embedding frame (rows/columns hidden around the visible image,
reflect padding) and cached. Reports the zero-shot score of the initial network, then the out-of-fold mIoU after
fine-tuning at several step counts. Saves the out-of-fold probabilities and a model trained on all images.
Usage: python ift.py --method trustmark --init out/trustmark_i1_last.pt --frames <json> [--lr_pad 4] --name ft1
frames json: {image_name: {"a": rows above, "b": rows below, ["l": cols left, "r": cols right]}} or {name: [a, b, l, r]}
"""
import argparse, copy, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.utils import miou_image
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

OUT = os.environ.get("INF_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))


def soft_miou_loss(logit, y):
    p = torch.sigmoid(logit.float()); d = (1, 2)
    i1 = (p * y).sum(d); u1 = (p + y - p * y).sum(d); q, z = 1 - p, 1 - y
    i0 = (q * z).sum(d); u0 = (q + z - q * z).sum(d)
    return 1 - 0.5 * ((i1 + 1) / (u1 + 1) + (i0 + 1) / (u0 + 1)).mean()


def template_on_frame(templ, x, bits, fr):
    a, b, l, r = [int(v) for v in fr]
    xp, ra, rb, rl, rr = x, a, b, l, r
    while ra or rb or rl or rr:                      # reflect padding larger than the image: repeat (same as numpy "reflect")
        H, W = xp.shape[-2:]; pa, pb, pl, pr = min(ra, H - 1), min(rb, H - 1), min(rl, W - 1), min(rr, W - 1)
        xp = F.pad(xp, (pl, pr, pa, pb), mode="reflect"); ra, rb, rl, rr = ra - pa, rb - pb, rl - pl, rr - pr
    return templ(xp, bits)[:, :, a:a + 512, l:l + 512]


def aug(x, t, y, rng):
    if rng.random() < 0.5: x, t, y = x.flip(-1), t.flip(-1), y.flip(-1)
    if rng.random() < 0.5: x, t, y = x.flip(-2), t.flip(-2), y.flip(-2)
    k = int(rng.integers(4))
    return torch.rot90(x, k, (-2, -1)), torch.rot90(t, k, (-2, -1)), torch.rot90(y, k, (-2, -1))


@torch.no_grad()
def predict(net, X, T, idx):
    net.eval(); out = []
    for i in range(0, len(idx), 16):
        j = idx[i:i + 16]
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(net(X[j].float(), T[j].float())[0].float()).cpu().numpy())
    return np.concatenate(out)


def score(P, G):
    s05 = float(np.mean([miou_image(p > 0.5, g) for p, g in zip(P, G)]))
    best = max((float(np.mean([miou_image(p > t, g) for p, g in zip(P, G)])), float(t)) for t in np.arange(0.1, 0.91, 0.1))
    return s05, best


def train(net, X, T, Y, idx, steps, bs, lr, rng, ckpts, hook):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda")
    for step in range(1, steps + 1):
        net.train(); xs, ts, ys = [], [], []
        for k in rng.choice(idx, bs):
            x, t, y = aug(X[k].float(), T[k].float(), Y[k], rng); xs.append(x); ts.append(t); ys.append(y)
        x, t, y = torch.stack(xs), torch.stack(ts), torch.stack(ys)
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = net(x.contiguous(memory_format=torch.channels_last), t)
        loss = F.binary_cross_entropy_with_logits(lg.float(), y) + soft_miou_loss(lg, y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
        if step in ckpts:
            hook(step, net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--init", required=True); ap.add_argument("--frames", required=True)
    ap.add_argument("--lr_pad", type=int, default=0); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--name", default="ft1"); ap.add_argument("--zero_only", action="store_true")
    a = ap.parse_args()
    dev = "cuda"; torch.backends.cudnn.benchmark = True; rng = np.random.default_rng(0)
    fr = {}
    for f in a.frames.split(","):
        for k, v in json.load(open(f)).items():
            fr[k] = [v["a"], v["b"], v.get("l", a.lr_pad), v.get("r", a.lr_pad)] if isinstance(v, dict) else list(v)
    templ = Templater(load_watermark(a.method, device=dev)); ds = LocalisationDataset(RELEASE, "train", methods=[a.method])
    names, X, T, T0, Y = [], [], [], [], []
    for i, m in enumerate(ds.meta):
        if m["image_name"] not in fr or not m.get("full_watermark", False): continue
        it = ds[i]; x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].to(dev).float(); b = torch.from_numpy(it["message"][None].astype(np.float32)).to(dev)
        with torch.no_grad():
            T.append(normalise(template_on_frame(templ, x, b, fr[m["image_name"]]))[0][0].half()); T0.append(normalise(templ(x, b))[0][0].half())
        names.append(m["image_name"]); X.append(x[0].to(torch.uint8)); Y.append(torch.from_numpy(it["mask128"]).float().to(dev))
    X, T, T0, Y = torch.stack(X), torch.stack(T), torch.stack(T0), torch.stack(Y); G = Y.cpu().numpy(); N = len(names); allidx = np.arange(N)
    ones = float(np.mean([miou_image(np.ones_like(g), g) for g in G]))
    init = CorrNet().to(dev).to(memory_format=torch.channels_last); init.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    tag = f"{a.method}_{a.name}"
    for nm, TT in (("template on the visible frame  ", T0), ("template on the estimated frame", T)):
        P = predict(init, X, TT, allidx); s05, best = score(P, G)
        print(f"{tag} n={N} zero-shot, {nm}: prob wm {np.mean([p[g > 0].mean() for p, g in zip(P, G)]):.3f} | holes {np.nanmean([p[g == 0].mean() if (g == 0).any() else np.nan for p, g in zip(P, G)]):.3f} "
              f"| mIoU@0.5 {s05:.4f} | best thr {best[1]:.1f}: {best[0]:.4f} | all-ones {ones:.4f}", flush=True)
    if a.zero_only: return
    fold = np.array([int(n[:6], 16) % 5 for n in names]); ckpts = sorted({a.steps // 4, a.steps // 2, 3 * a.steps // 4, a.steps})
    OOF = {c: np.zeros((N, 128, 128), np.float32) for c in ckpts}; t0 = time.time()
    for f in range(5):
        tr, va = allidx[fold != f], allidx[fold == f]; net = copy.deepcopy(init)
        def hook(step, n, va=va):
            OOF[step][va] = predict(n, X, T, va)
        train(net, X, T, Y, tr, a.steps, a.bs, a.lr, rng, ckpts, hook)
        print(f"{tag} fold {f} done ({len(va)} held out) {time.time() - t0:.0f}s", flush=True)
    res = {}
    for c in ckpts:
        s05, best = score(OOF[c], G); res[c] = best[0]
        print(f"{tag} OUT-OF-FOLD after {c} steps: mIoU@0.5 {s05:.4f} | best thr {best[1]:.1f}: {best[0]:.4f} | all-ones {ones:.4f}", flush=True)
    cb = max(res, key=res.get)
    np.savez_compressed(f"{OUT}/{tag}_oof.npz", names=np.array(names), probs=OOF[cb].astype(np.float16), frames=np.array([fr[n] for n in names]))
    net = copy.deepcopy(init); train(net, X, T, Y, allidx, cb, a.bs, a.lr, rng, {cb}, lambda s, n: None)
    torch.save(net.state_dict(), f"{OUT}/{tag}_all.pt"); print("DONE", tag, "steps", cb, "saved", f"{OUT}/{tag}_oof.npz", f"{OUT}/{tag}_all.pt")


if __name__ == "__main__":
    main()
