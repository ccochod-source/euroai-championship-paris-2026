"""Train the template-informed localisation network for one watermark on synthetic fully-watermarked composites;
validate on the real train split (never used for training here).
Curriculum: the residual present in the image is amplified by alpha in [1, amax], amax decays to 1 during warm-up.
Usage: python itrain.py --method pixelseal --steps 12000 --name i1"""
import argparse, copy, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT as _OUT_OTHER, KIT  # noqa  (adds the kit to sys.path)
from idata import SynthFull, Real
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.utils import miou_image
from wmloc.watermarks import load_watermark

OUT = os.environ.get("INF_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))


def soft_miou_loss(logit, y):
    p = torch.sigmoid(logit.float()); y = y.float(); d = (1, 2)
    i1 = (p * y).sum(d); u1 = (p + y - p * y).sum(d); q, z = 1 - p, 1 - y
    i0 = (q * z).sum(d); u0 = (q + z - q * z).sum(d)
    return 1 - 0.5 * ((i1 + 1) / (u1 + 1) + (i0 + 1) / (u0 + 1)).mean()


@torch.no_grad()
def evaluate(net, templ, loader, dev):
    net.eval(); rows = []; probs = {}
    for x, m, bits, names, full in loader:
        x = x.to(dev).float(); t, _ = normalise(templ(x, bits.to(dev)))
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = net(x.contiguous(memory_format=torch.channels_last), t)
        p = torch.sigmoid(lg.float()).cpu().numpy()
        for k in range(len(names)):
            rows.append((miou_image(p[k] > 0.5, m[k].numpy()), bool(full[k]))); probs[names[k]] = p[k].astype(np.float16)
    s = np.array([r[0] for r in rows]); f = np.array([r[1] for r in rows])
    return float(s.mean()), float(s[f].mean()) if f.any() else 0.0, float(s[~f].mean()) if (~f).any() else 0.0, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--name", default="i1"); ap.add_argument("--init", default="")
    ap.add_argument("--amax", type=float, default=6.0); ap.add_argument("--warm", type=float, default=0.35)
    ap.add_argument("--aux", type=float, default=0.5); ap.add_argument("--max_minutes", type=float, default=1e9)
    a = ap.parse_args()
    dev = "cuda"; torch.backends.cudnn.benchmark = True
    os.makedirs(OUT, exist_ok=True); tag = f"{a.method}_{a.name}"
    templ = Templater(load_watermark(a.method, device=dev))
    dl = DataLoader(SynthFull(a.method), batch_size=a.bs, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True,
                    prefetch_factor=4, sampler=torch.utils.data.RandomSampler(range(10 ** 9), replacement=True, num_samples=a.steps * a.bs))
    val = DataLoader(Real("train", a.method), batch_size=16, num_workers=4)
    net = CorrNet().to(dev).to(memory_format=torch.channels_last)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    ema = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda"); best = -1; t0 = time.time(); run = np.zeros(3); log = []
    print(tag, "params %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6), flush=True)
    for step, (x, y, bits, resid) in enumerate(dl, 1):
        net.train()
        x = x.to(dev, non_blocking=True).float(); y = y.to(dev, non_blocking=True); resid = resid.to(dev, non_blocking=True).float()
        amax = 1 + (a.amax - 1) * max(0.0, 1 - step / (a.warm * a.steps))
        alpha = 1 + (amax - 1) * torch.rand(len(x), 1, 1, 1, device=dev)
        x = (x + (alpha - 1) * resid).clamp(0, 255).round()
        t, s = normalise(templ(x, bits.to(dev)))
        with torch.autocast("cuda", dtype=torch.float16):
            lg, E = net(x.contiguous(memory_format=torch.channels_last), t)
        bce = F.binary_cross_entropy_with_logits(lg.float(), y); iou = soft_miou_loss(lg, y)
        aux = F.smooth_l1_loss(E.float(), alpha * resid / s)
        loss = bce + iou + a.aux * aux
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
        d = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn, 1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        run += [bce.item(), iou.item(), aux.item()]
        if step % 100 == 0:
            print(f"{tag} step {step}/{a.steps} bce {run[0]/100:.4f} iouloss {run[1]/100:.4f} aux {run[2]/100:.4f} amax {amax:.2f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True); run[:] = 0
        stop = (time.time() - t0) / 60 > a.max_minutes
        if step % a.eval_every == 0 or step == a.steps or stop:
            sc, sf, sn, probs = evaluate(ema, templ, val, dev)
            log.append(dict(step=step, score=sc, full=sf, notfull=sn, amax=amax)); json.dump(log, open(f"{OUT}/{tag}_log.json", "w"))
            print(f"{tag} EVAL step {step} real-train mIoU FULL {sf:.4f} (notfull {sn:.4f}, all {sc:.4f}) best full {max(best, sf):.4f}", flush=True)
            torch.save(ema.state_dict(), f"{OUT}/{tag}_last.pt")
            if sf > best:
                best = sf; torch.save(ema.state_dict(), f"{OUT}/{tag}_best.pt"); np.savez_compressed(f"{OUT}/{tag}_valprobs.npz", **probs)
        if stop:
            print("TIME_LIMIT reached", flush=True); break
    print("TRAIN_DONE", tag, "best full", best)


if __name__ == "__main__":
    main()
