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


def mosaic(X, M, pool, rng):
    """One training image made of 4 quadrants, each a random-offset crop of a different real image (offsets are
    multiples of 4 px so labels stay on the 128 grid). Real residuals, new layouts, no absolute-position shortcut."""
    cy, cx = [int(v) for v in rng.integers(8, 121, 2)]            # split point on the 128 grid
    x = torch.empty(3, 512, 512, dtype=X.dtype); m = torch.empty(128, 128, dtype=M.dtype)
    for (y0, y1), (x0, x1) in (((0, cy), (0, cx)), ((0, cy), (cx, 128)), ((cy, 128), (0, cx)), ((cy, 128), (cx, 128))):
        i = int(rng.choice(pool)); h, w = y1 - y0, x1 - x0
        if rng.random() < 0.5: sy, sx = y0, x0                    # keep the original position
        else: sy, sx = int(rng.integers(0, 128 - h + 1)), int(rng.integers(0, 128 - w + 1))
        x[:, y0 * 4:y1 * 4, x0 * 4:x1 * 4] = X[i, :, sy * 4:(sy + h) * 4, sx * 4:(sx + w) * 4]
        m[y0:y1, x0:x1] = M[i, sy:sy + h, sx:sx + w]
    return x, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--init", default="v1"); ap.add_argument("--name", default="ft1")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--init_ckpt", default=""); ap.add_argument("--hpf", type=int, default=0); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--real_bs", type=int, default=8); ap.add_argument("--syn_bs", type=int, default=8)
    ap.add_argument("--cutmix", type=int, default=1); ap.add_argument("--p_in", type=float, default=0.0); ap.add_argument("--full_fit", type=int, default=0); ap.add_argument("--mosaic", type=float, default=0.0); ap.add_argument("--tags", default="a"); ap.add_argument("--pseudo", default=""); ap.add_argument("--pseudo_thr", type=float, default=0.5)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    X, M, names, full = load_real("train", a.method); Xt, _, tnames, _ = load_real("test", a.method)
    rng = np.random.default_rng(0); fold = np.zeros(len(X), int)
    for v in (True, False):
        idx = np.where(full == v)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % a.folds
    NR = len(X)                                   # real labelled images are indices [0, NR); pseudo-labelled test images follow
    if a.pseudo:
        z = np.load(f"{OUT}/{a.method}_{a.pseudo}_testprobs.npz"); pp = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
        Mt = torch.from_numpy(np.stack([pp[n] for n in tnames]) > a.pseudo_thr).float()
        X = torch.cat([X, Xt]); M = torch.cat([M, Mt]); print("pseudo-labelled test images added:", len(Xt), flush=True)
    extra = np.arange(NR, len(X))
    syn = iter(DataLoader(Synth(a.method, tags=tuple(a.tags.split(",")), n=10 ** 9, p_in=a.p_in), batch_size=a.syn_bs, num_workers=16, drop_last=True,
                          sampler=torch.utils.data.RandomSampler(range(10 ** 6), replacement=True, num_samples=10 ** 8))) if a.syn_bs > 0 else None
    init = torch.load(a.init_ckpt or f"{OUT}/{a.method}_{a.init}_best.pt", map_location="cpu", weights_only=True)
    oof = np.zeros((len(fold), 128, 128), np.float32); test = np.zeros((len(Xt), 128, 128), np.float32)
    base = Net(hpf=bool(a.hpf)).cuda().to(memory_format=torch.channels_last); base.load_state_dict(init)
    p0 = predict(base, X[:NR]); print(a.method, "before fine-tuning:", np.mean([miou_image(p0[i] > 0.5, M[i].numpy()) for i in range(NR)]), flush=True)
    g = torch.Generator().manual_seed(0)
    todo = [-1] if a.full_fit else list(range(a.folds))
    for k in todo:
        tr = np.concatenate([np.where(fold != k)[0], extra]); va = np.where(fold == k)[0]
        net = Net(hpf=bool(a.hpf)).cuda().to(memory_format=torch.channels_last); net.load_state_dict(init)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda")
        for step in range(1, a.steps + 1):
            net.train()
            bi = torch.from_numpy(rng.choice(tr, a.real_bs, replace=False)); xr, mr = X[bi].clone(), M[bi].clone()
            for q in range(a.real_bs):
                if rng.random() < a.mosaic: xr[q], mr[q] = mosaic(X, M, tr, rng)
            if a.cutmix: xr, mr = cutmix(xr, mr, g)
            if syn is not None:
                xs, ms = next(syn); xr = torch.cat([xr, xs]); mr = torch.cat([mr, ms])
            x = xr.cuda().contiguous(memory_format=torch.channels_last); y = mr.cuda()
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
        s = np.array([miou_image(oof[i] > 0.5, M[i].numpy()) for i in range(NR)])
        print(f"FT_DONE {a.method} OOF mIoU {s.mean():.4f} (full {s[full].mean():.4f} / notfull {s[~full].mean():.4f})", flush=True)
        np.savez_compressed(f"{OUT}/{a.method}_{a.name}_trainprobs.npz", names=np.array(names), probs=oof.astype(np.float16))
    np.savez_compressed(f"{OUT}/{a.method}_{a.name}_testprobs.npz", names=np.array(tnames), probs=test.astype(np.float16))


if __name__ == "__main__":
    main()
