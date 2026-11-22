"""K-fold fine-tuning on the real train split (domain alignment), starting from the synthetic-trained network.
Each batch mixes real images (from the training folds, with CutMix between real images) and synthetic composites.
Writes out-of-fold probabilities for the real train split and fold-averaged probabilities for the test split."""
import argparse, copy, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from common import OUT
from data import Synth, Real
from model import Net
from train import soft_miou_loss
from wmloc.utils import miou_image


def load_real(split, method):
    xs, ms, names, fulls = [], [], [], []
    for x, m, n, f in DataLoader(Real(split, method), batch_size=32, num_workers=8):
        xs.append(x); ms.append(m); names += list(n); fulls += [bool(v) for v in f]
    return torch.cat(xs), torch.cat(ms), names, np.array(fulls)


@torch.no_grad()
def predict(net, x):
    net.eval(); out = []
    for i in range(0, len(x), 16):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(net(x[i:i + 16].cuda().contiguous(memory_format=torch.channels_last)).float()).cpu())
    return torch.cat(out).numpy()


def cutmix(x, m, g):
    """Paste a random rectangle (aligned to the 4-px label grid) from a permuted batch: real residuals, new layouts."""
    x, m = x.clone(), m.clone(); perm = torch.randperm(len(x), generator=g)
    for i in range(len(x)):
        if torch.rand(1, generator=g).item() < 0.5:
            h, w = [int(v) for v in torch.randint(8, 96, (2,), generator=g)]
            y0 = int(torch.randint(0, 128 - h + 1, (1,), generator=g)); x0 = int(torch.randint(0, 128 - w + 1, (1,), generator=g))
            x[i, :, y0 * 4:(y0 + h) * 4, x0 * 4:(x0 + w) * 4] = x[perm[i], :, y0 * 4:(y0 + h) * 4, x0 * 4:(x0 + w) * 4]
            m[i, y0:y0 + h, x0:x0 + w] = m[perm[i], y0:y0 + h, x0:x0 + w]
    return x, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--init", default="v1"); ap.add_argument("--name", default="ft1")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--real_bs", type=int, default=8); ap.add_argument("--syn_bs", type=int, default=8)
    ap.add_argument("--cutmix", type=int, default=1); ap.add_argument("--p_in", type=float, default=0.0); ap.add_argument("--full_fit", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    X, M, names, full = load_real("train", a.method); Xt, _, tnames, _ = load_real("test", a.method)
    rng = np.random.default_rng(0); fold = np.zeros(len(X), int)
    for v in (True, False):
        idx = np.where(full == v)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % a.folds
    syn = iter(DataLoader(Synth(a.method, n=10 ** 9, p_in=a.p_in), batch_size=a.syn_bs, num_workers=16, drop_last=True,
                          sampler=torch.utils.data.RandomSampler(range(10 ** 6), replacement=True, num_samples=10 ** 8)))
    init = torch.load(f"{OUT}/{a.method}_{a.init}_best.pt", map_location="cpu", weights_only=True)
    oof = np.zeros((len(X), 128, 128), np.float32); test = np.zeros((len(Xt), 128, 128), np.float32)
    base = Net().cuda().to(memory_format=torch.channels_last); base.load_state_dict(init)
    p0 = predict(base, X); print(a.method, "before fine-tuning:", np.mean([miou_image(p0[i] > 0.5, M[i].numpy()) for i in range(len(X))]), flush=True)
    g = torch.Generator().manual_seed(0)
    todo = [-1] if a.full_fit else list(range(a.folds))
    for k in todo:
        tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
        net = Net().cuda().to(memory_format=torch.channels_last); net.load_state_dict(init)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda")
        for step in range(1, a.steps + 1):
            net.train()
            bi = torch.from_numpy(rng.choice(tr, a.real_bs, replace=False)); xr, mr = X[bi], M[bi]
            if a.cutmix: xr, mr = cutmix(xr, mr, g)
            xs, ms = next(syn)
            x = torch.cat([xr, xs]).cuda().contiguous(memory_format=torch.channels_last); y = torch.cat([mr, ms]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                lg = net(x)
            loss = F.binary_cross_entropy_with_logits(lg.float(), y) + soft_miou_loss(lg, y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
            if step % 500 == 0 and len(va):
                p = predict(net, X[va]); print(f"{a.method} fold {k} step {step} oof mIoU {np.mean([miou_image(p[i] > 0.5, M[va[i]].numpy()) for i in range(len(va))]):.4f}", flush=True)
        if len(va): oof[va] = predict(net, X[va])
        test += predict(net, Xt) / len(todo)
        torch.save(net.state_dict(), f"{OUT}/{a.method}_{a.name}_fold{k}.pt")
    if not a.full_fit:
        s = np.array([miou_image(oof[i] > 0.5, M[i].numpy()) for i in range(len(X))])
        print(f"FT_DONE {a.method} OOF mIoU {s.mean():.4f} (full {s[full].mean():.4f} / notfull {s[~full].mean():.4f})", flush=True)
        np.savez_compressed(f"{OUT}/{a.method}_{a.name}_trainprobs.npz", names=np.array(names), probs=oof.astype(np.float16))
    np.savez_compressed(f"{OUT}/{a.method}_{a.name}_testprobs.npz", names=np.array(tnames), probs=test.astype(np.float16))


if __name__ == "__main__":
    main()
