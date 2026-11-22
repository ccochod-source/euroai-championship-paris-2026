"""Train one localisation network for one watermark method on synthetic composites; validate on the real train split.
Amplitude curriculum: the watermark residual is amplified by a factor alpha (per sample, log-uniform in
[1, alpha_max]); alpha_max starts high so the network finds the signal, then decays to 1 (the real strength)."""
import argparse, copy, json, math, os, time, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from common import OUT
from data import Synth, Real
from model import Net
from bands import blur
from wmloc.utils import miou_image


def soft_miou_loss(logit, y):
    p = torch.sigmoid(logit.float()); y = y.float(); d = (1, 2)
    i1 = (p * y).sum(d); u1 = (p + y - p * y).sum(d)
    q, z = 1 - p, 1 - y
    i0 = (q * z).sum(d); u0 = (q + z - q * z).sum(d)
    return 1 - 0.5 * ((i1 + 1) / (u1 + 1) + (i0 + 1) / (u0 + 1)).mean()


@torch.no_grad()
def evaluate(net, loader, dev):
    net.eval(); rows = []; probs = {}
    for x, m, names, full in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)).float()).cpu().numpy()
        for k in range(len(names)):
            rows.append((miou_image(p[k] > 0.5, m[k].numpy()), bool(full[k]))); probs[names[k]] = p[k]
    s = np.array([r[0] for r in rows]); f = np.array([r[1] for r in rows])
    return float(s.mean()), float(s[f].mean()) if f.any() else 0.0, float(s[~f].mean()) if (~f).any() else 0.0, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--tags", default="a")
    ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--name", default="v1")
    ap.add_argument("--init", default=""); ap.add_argument("--p_in", type=float, default=0.0)
    ap.add_argument("--hpf", type=int, default=1); ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--bce_thr", type=float, default=0.30); ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--iou_w", type=float, default=1.0)
    ap.add_argument("--res_aug", type=int, default=0, help="residual augmentation: random blur and amplitude 0.6-1.0, to cover the real (smoother, weaker) watermark")
    a = ap.parse_args()
    dev = "cuda"; torch.backends.cudnn.benchmark = True
    os.makedirs(OUT, exist_ok=True); tag = f"{a.method}_{a.name}"
    ds = Synth(a.method, tags=tuple(a.tags.split(",")), n=10 ** 9, p_in=a.p_in, pairs=True)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                    sampler=torch.utils.data.RandomSampler(ds, replacement=True, num_samples=a.steps * a.bs))
    val = DataLoader(Real("train", a.method), batch_size=16, num_workers=4)
    net = Net(hpf=bool(a.hpf)).to(dev).to(memory_format=torch.channels_last)
    if a.init: net.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    ema = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda"); best = -1; t = time.time(); run = [0.0, 0.0]; log = []; amax = a.alpha0
    json.dump(vars(a), open(f"{OUT}/{tag}_args.json", "w"))
    print(tag, "params %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6), vars(a), flush=True)
    for step, (c, w, m, y) in enumerate(dl, 1):
        net.train()
        c = c.to(dev, non_blocking=True); w = w.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True)[:, None]; y = y.to(dev, non_blocking=True)
        if amax > 1.0001 or a.res_aug:
            B = c.shape[0]; cf = c.float(); r = w.float() - cf
            al = (max(amax, 1.0) * torch.exp(-torch.rand(B, 1, 1, 1, device=dev) * math.log(3.0))).clamp_(min=1.0)   # log-uniform in [amax/3, amax]
            if a.res_aug:
                g = torch.randint(0, 4, (B,), device=dev)                       # 0,1: untouched  2: blur 0.5  3: blur 0.8
                for gi, s in ((2, 0.5), (3, 0.8)):
                    k = g == gi
                    if k.any(): r[k] = blur(r[k], s)
                al = al * (0.6 + 0.4 * torch.rand(B, 1, 1, 1, device=dev))
            x = torch.where(m, (cf + al * r).clamp_(0, 255).round_(), cf)
        else:
            x = torch.where(m, w, c)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = net(x)
        bce = F.binary_cross_entropy_with_logits(lg.float(), y); iou = soft_miou_loss(lg, y); loss = bce + a.iou_w * iou
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
        d = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn, 1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        run[0] += bce.item(); run[1] += iou.item()
        if step % 100 == 0:
            print(f"{tag} step {step}/{a.steps} bce {run[0]/100:.4f} iouloss {run[1]/100:.4f} amax {amax:.2f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t:.0f}s", flush=True)
            b_ = run[0] / 100                                  # hold the amplification until the signal is caught, then decay
            if amax > 1.0 and b_ < 0.60: amax = max(1.0, amax * (0.90 if b_ < a.bce_thr else 0.96))
            run = [0.0, 0.0]
        stop = (time.time() - t) / 60 > a.max_minutes
        if step % a.eval_every == 0 or step == a.steps or stop:
            s, sf, sn, probs = evaluate(ema, val, dev)
            log.append(dict(step=step, score=s, full=sf, notfull=sn, amax=amax)); json.dump(log, open(f"{OUT}/{tag}_log.json", "w"))
            print(f"{tag} EVAL step {step} real-train mIoU {s:.4f} (full {sf:.4f} / notfull {sn:.4f}) best {max(best, s):.4f} amax {amax:.2f}", flush=True)
            torch.save(ema.state_dict(), f"{OUT}/{tag}_last.pt"); torch.save(net.state_dict(), f"{OUT}/{tag}_lastraw.pt")
            if s > best:
                best = s; torch.save(ema.state_dict(), f"{OUT}/{tag}_best.pt"); np.savez_compressed(f"{OUT}/{tag}_valprobs.npz", **probs)
        if stop: print("stopping on time limit"); break
    print("TRAIN_DONE", tag, "best", best)


if __name__ == "__main__":
    main()
