"""Informed localisation network, version 2: synthetic data that follows the pipeline found on the real images.
Real fully watermarked images are (resized) square crops of a larger watermarked frame: the kit template only matches
when computed on that frame. Training sample: kit-embedded 512 image composited with clean holes, random square crop
(side 464-512) at a random offset, resized to 512; the template is computed on the reflect-padded canvas with the true
margins (+- jitter) and cropped back, exactly as at test time (frame4.py).
Optional real images (--real_frames): real train images of the method with their estimated frames are mixed into the
batches (fold --fold of 5 is held out and used for validation).
Usage: python itrain2.py --method stegastamp --init out/stegastamp_i1_last.pt --steps 5000 --name i2"""
import argparse, copy, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import KIT, SYN, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.utils import miou_image
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

OUT = os.environ.get("INF_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))


def soft_miou_loss(logit, y):
    p = torch.sigmoid(logit.float()); y = y.float(); d = (1, 2)
    i1 = (p * y).sum(d); u1 = (p + y - p * y).sum(d); q, z = 1 - p, 1 - y
    i0 = (q * z).sum(d); u0 = (q + z - q * z).sum(d)
    return 1 - 0.5 * ((i1 + 1) / (u1 + 1) + (i0 + 1) / (u0 + 1)).mean()


def template_on_frame(templ, x, bits, fr):
    """x: 1 x 3 x 512 x 512 [0,255]; fr = (above, below, left, right). Template on the reflect-padded canvas, cropped back."""
    a, b, l, r = [int(v) for v in fr]
    xp, ra, rb, rl, rr = x, a, b, l, r
    while ra or rb or rl or rr:
        H, W = xp.shape[-2:]; pa, pb, pl, pr = min(ra, H - 1), min(rb, H - 1), min(rl, W - 1), min(rr, W - 1)
        xp = F.pad(xp, (pl, pr, pa, pb), mode="reflect"); ra, rb, rl, rr = ra - pa, rb - pb, rl - pl, rr - pr
    return templ(xp, bits)[:, :, a:a + 512, l:l + 512]


class FluxCanvas(Dataset):
    """Clean 768 x 768 image (FLUX 1024 original, downscaled) + a 512 x 512 hole mask (True = stays watermarked)."""
    def __init__(self, n=10 ** 9, seed=0, p_allwm=0.05):
        import glob, pickle
        from common import CLEAN
        self.files = sorted(glob.glob(f"{CLEAN}/flux1024/**/*.jpg", recursive=True) + glob.glob(f"{CLEAN}/flux1024/**/*.png", recursive=True))
        # shape_bank.pkl is our own file (written by gen_masks.py from COCO annotations), not third-party data
        self.bank = pickle.load(open(f"{SYN}/shape_bank.pkl", "rb")); self.n, self.seed, self.p_allwm = n, seed, p_allwm

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from data import make_mask
        from PIL import Image
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, i, info.id if info else 0, int(torch.randint(0, 2 ** 31 - 1, (1,)))])
        im = Image.open(self.files[int(rng.integers(len(self.files)))]).convert("RGB")
        if rng.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = np.asarray(im.resize((768, 768), Image.Resampling.LANCZOS))
        m = np.ones((512, 512), bool) if rng.random() < self.p_allwm else make_mask(rng, self.bank, (0, 0, 0, 0))[0]
        return torch.from_numpy(im.copy()).permute(2, 0, 1), torch.from_numpy(m.astype(np.uint8))


def make_batch(templ, big, m0, nbits, rng, dev, emax=160, jit=2):
    """Pipeline of the real data: the kit embeds a square frame of side 512 + e, the image is a 512 crop of it at a free
    offset, clean holes are pasted. Template: computed on the visible image padded by reflection to the frame (+- jitter)."""
    X, T, Y = [], [], []
    for i in range(len(big)):
        e = 0 if rng.random() < 0.1 else int(rng.integers(1, emax + 1)); S = 512 + e
        oy, ox = int(rng.integers(0, 768 - S + 1)), int(rng.integers(0, 768 - S + 1))
        C = big[i:i + 1, :, oy:oy + S, ox:ox + S].float(); bits = torch.randint(0, 2, (1, nbits), device=dev).float()
        r = templ(C, bits); q = rng.random()
        W = (C + r).floor() if q < 0.4 else (C + r).round()            # the kit truncates (TrustMark, StegaStamp) or rounds (PixelSeal)
        W = W.clamp(0, 255)
        a, l = int(rng.integers(0, e + 1)), int(rng.integers(0, e + 1))
        m = m0[i:i + 1, None].bool()
        x = torch.where(m, W[:, :, a:a + 512, l:l + 512], C[:, :, a:a + 512, l:l + 512])
        fr = [max(0, v + int(rng.integers(-jit, jit + 1))) for v in (a, e - a, l, e - l)]
        X.append(x); T.append(template_on_frame(templ, x, bits, fr)); Y.append((F.avg_pool2d(m.float(), 4)[:, 0] >= 0.5).float())
    return torch.cat(X), normalise(torch.cat(T))[0], torch.cat(Y)


class RealSet:
    """Real train images of one method with their estimated frames; templates are cached on the GPU."""
    def __init__(self, method, frames, templ, dev, fold=-1):
        ds = LocalisationDataset(RELEASE, "train", methods=[method])
        self.tr, self.va = [], []
        for i, m in enumerate(ds.meta):
            nm = m["image_name"]
            if nm not in frames: continue
            it = ds[i]; x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].to(dev).float()
            b = torch.from_numpy(it["message"][None].astype(np.float32)).to(dev)
            with torch.no_grad():
                t = normalise(template_on_frame(templ, x, b, frames[nm]))[0]
            rec = (x[0].to(torch.uint8), t[0].half(), torch.from_numpy(it["mask128"]).float().to(dev), nm, bool(m.get("full_watermark", False)))
            (self.va if (fold >= 0 and int(nm[:6], 16) % 5 == fold) else self.tr).append(rec)

    def sample(self, n, rng):
        X, T, Y = [], [], []
        for k in rng.integers(0, len(self.tr), n):
            x, t, y, _, _ = self.tr[int(k)]; x = x.float(); t = t.float()
            if rng.random() < 0.5: x, t, y = x.flip(-1), t.flip(-1), y.flip(-1)      # the same geometric transform on image, template
            if rng.random() < 0.5: x, t, y = x.flip(-2), t.flip(-2), y.flip(-2)      # and mask keeps their relation intact
            k4 = int(rng.integers(4)); x, t, y = torch.rot90(x, k4, (-2, -1)), torch.rot90(t, k4, (-2, -1)), torch.rot90(y, k4, (-2, -1))
            X.append(x); T.append(t); Y.append(y)
        return torch.stack(X), torch.stack(T), torch.stack(Y)


@torch.no_grad()
def eval_real(net, recs):
    net.eval(); out = []
    for i in range(0, len(recs), 16):
        x = torch.stack([r[0] for r in recs[i:i + 16]]).float(); t = torch.stack([r[1] for r in recs[i:i + 16]]).float()
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(x, t)[0].float())
        for r, pp in zip(recs[i:i + 16], p):
            out.append((r[3], r[4], pp.cpu().numpy(), r[2].cpu().numpy()))
    return out


def summarise(res):
    best = (0, 0.5)
    for thr in np.arange(0.1, 0.91, 0.1):
        s = np.mean([miou_image(p > thr, g) for _, f, p, g in res if f]) if any(r[1] for r in res) else 0
        if s > best[0]: best = (float(s), float(thr))
    f05 = float(np.mean([miou_image(p > 0.5, g) for _, f, p, g in res if f])) if any(r[1] for r in res) else 0.0
    ones = float(np.mean([miou_image(np.ones_like(g), g) for _, f, p, g in res if f])) if any(r[1] for r in res) else 0.0
    return f05, best, ones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--name", default="i2"); ap.add_argument("--init", default=""); ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--real_frames", default=""); ap.add_argument("--fold", type=int, default=-1); ap.add_argument("--n_real", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"; torch.backends.cudnn.benchmark = True; rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True); tag = f"{a.method}_{a.name}"
    templ = Templater(load_watermark(a.method, device=dev))
    dl = DataLoader(FluxCanvas(), batch_size=a.bs - a.n_real, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True,
                    prefetch_factor=4, sampler=torch.utils.data.RandomSampler(range(10 ** 9), replacement=True, num_samples=a.steps * a.bs))
    real = None
    if a.real_frames:
        fr = {}
        for f in a.real_frames.split(","):
            for k, v in json.load(open(f)).items():
                if isinstance(v, dict):
                    if v.get("slope", 1) >= 0.2: fr[k] = [v["a"], v["b"], v.get("l", 0), v.get("r", 0)]
                else: fr[k] = list(v)
        real = RealSet(a.method, fr, templ, dev, a.fold)
        print(tag, "real images with a frame: train", len(real.tr), "held out", len(real.va), flush=True)
    net = CorrNet().to(dev).to(memory_format=torch.channels_last)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    ema = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda"); t0 = time.time(); run = np.zeros(2); log = []; best = -1
    for step, (x0, m0) in enumerate(dl, 1):
        net.train()
        x, t, y = make_batch(templ, x0.to(dev), m0.to(dev), templ.wm.nbits, rng, dev)
        if real is not None and a.n_real:
            xr, tr, yr = real.sample(a.n_real, rng); x, t, y = torch.cat([x, xr]), torch.cat([t, tr]), torch.cat([y, yr])
        with torch.autocast("cuda", dtype=torch.float16):
            lg, _ = net(x.contiguous(memory_format=torch.channels_last), t)
        bce = F.binary_cross_entropy_with_logits(lg.float(), y); iou = soft_miou_loss(lg, y); loss = bce + iou
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
        d = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn, 1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        run += [bce.item(), iou.item()]
        if step % 100 == 0:
            print(f"{tag} step {step}/{a.steps} bce {run[0]/100:.4f} iouloss {run[1]/100:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True); run[:] = 0
        stop = (time.time() - t0) / 60 > a.max_minutes
        if step % a.eval_every == 0 or step == a.steps or stop:
            torch.save(ema.state_dict(), f"{OUT}/{tag}_last.pt")
            if real is not None:
                recs = real.va if real.va else real.tr
                f05, bst, ones = summarise(eval_real(ema, recs)); log.append(dict(step=step, full05=f05, best=bst, ones=ones))
                print(f"{tag} EVAL step {step} real {'held-out' if real.va else 'train'} FULL mIoU@0.5 {f05:.4f}  best thr {bst[1]:.1f}: {bst[0]:.4f}  (all-ones {ones:.4f})", flush=True)
                if bst[0] > best:
                    best = bst[0]; torch.save(ema.state_dict(), f"{OUT}/{tag}_best.pt")
                json.dump(log, open(f"{OUT}/{tag}_log.json", "w"))
        if stop:
            print("TIME_LIMIT reached", flush=True); break
    print("TRAIN_DONE", tag)


if __name__ == "__main__":
    main()
