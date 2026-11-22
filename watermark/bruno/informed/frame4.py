"""Estimate the embedding frame of a fully watermarked REAL image as 4 margins around the visible 512 x 512 image
(extra pixels above, below, left, right): the kit template is computed on the reflect-padded canvas and cropped back;
the informed detector response (mean probability) is maximised. Multi-start on a coarse grid, then coordinate-wise
hill climbing with steps 8, 4, 2, 1.
Output npz: names, probs (128 x 128 at the best frame), probs_visible, frames (above, below, left, right), score.
Usage: python frame4.py --method stegastamp --split train --ckpt out/stegastamp_i1_last.pt --out out/f4_stegastamp_train.npz
"""
import argparse, os, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image

P = 256


class Finder:
    def __init__(self, method, ckpt, dev="cuda"):
        self.wm = load_watermark(method, device=dev); self.templ = Templater(self.wm)
        self.net = CorrNet().to(dev).eval(); self.net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.n_eval = 0

    @torch.no_grad()
    def detect(self, x, xp, bits, frames, bs=12):
        """frames: list of (above, below, left, right). Returns N x 128 x 128 probabilities."""
        out = []
        for i in range(0, len(frames), bs):
            Ts = []
            for a, b, l, r in frames[i:i + bs]:
                T = self.templ(xp[:, :, P - a:P + 512 + b, P - l:P + 512 + r], bits)
                Ts.append(T[:, :, a:a + 512, l:l + 512])
            t, _ = normalise(torch.cat(Ts)); self.n_eval += len(Ts)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(torch.sigmoid(self.net(x.expand(len(Ts), -1, -1, -1), t)[0].float()))
        return torch.cat(out)

    def search(self, img, msg, emax=48, max_m=P, gstep=4, ntop=3):
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].cuda().float()
        xp = F.pad(x, (P, P, P, P), mode="reflect"); bits = torch.from_numpy(msg[None].astype(np.float32)).cuda()
        # the frames found so far are squares of side 512 + e (e up to ~30 px) with a free offset: dense grid on (e, a, l)
        starts = [(a, e - a, l, e - l) for e in range(0, emax + 1, gstep) for a in range(0, e + 1, gstep) for l in range(0, e + 1, gstep)]
        sc = self.detect(x, xp, bits, starts).mean((1, 2)).cpu().numpy()
        best = None
        for k in np.argsort(-sc)[:ntop]:                      # hill climb from the 3 best starts
            f, ps = starts[int(k)], float(sc[int(k)]); step = 8
            while step >= 1:
                nb = []
                for d in range(4):
                    for sgn in (-step, step):
                        q = list(f); q[d] += sgn
                        if 0 <= q[d] <= max_m: nb.append(tuple(q))
                for d0, d1 in ((0, 1), (2, 3)):            # shift the frame without changing its size
                    for sgn in (-step, step):
                        q = list(f); q[d0] += sgn; q[d1] -= sgn
                        if min(q[d0], q[d1]) >= 0 and max(q[d0], q[d1]) <= max_m: nb.append(tuple(q))
                sn = self.detect(x, xp, bits, nb).mean((1, 2)).cpu().numpy(); j = int(sn.argmax())
                if sn[j] > ps + 1e-4:
                    f, ps = nb[j], float(sn[j])
                else:
                    step //= 2
            if best is None or ps > best[1]:
                best = (f, ps)
        pb = self.detect(x, xp, bits, [best[0], (0, 0, 0, 0)])
        return dict(frame=best[0], score=best[1], prob=pb[0].cpu().numpy(), prob_visible=pb[1].cpu().numpy(), start_best=float(sc.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--acc_thr", type=float, default=0.75); ap.add_argument("--emax", type=int, default=48); ap.add_argument("--gstep", type=int, default=4); ap.add_argument("--ntop", type=int, default=3)
    a = ap.parse_args()
    fd = Finder(a.method, a.ckpt); ds = LocalisationDataset(RELEASE, a.split, methods=[a.method])
    k0, kn = [int(v) for v in a.shard.split("/")]; idx = list(range(len(ds)))[k0::kn]
    names, probs, probs0, frames, scores, rows, t0 = [], [], [], [], [], [], time.time()
    for i in idx:
        it = ds[i]
        x1 = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).cuda()
        acc = float(((fd.wm.logits(x1)[0].float().cpu().numpy() > 0) == (it["message"] > 0)).mean())
        if acc <= a.acc_thr:
            continue
        n0 = fd.n_eval; r = fd.search(it["image"], it["message"], emax=a.emax, gstep=a.gstep, ntop=a.ntop)
        names.append(it["image_name"]); probs.append(r["prob"].astype(np.float16)); probs0.append(r["prob_visible"].astype(np.float16)); frames.append(r["frame"]); scores.append(r["score"])
        if "mask128" in it:
            g = it["mask128"]
            rows.append((r["prob_visible"][g > 0].mean(), r["prob"][g > 0].mean(), r["prob"][g == 0].mean() if (g == 0).any() else np.nan,
                         miou_image(r["prob_visible"] > 0.5, g), miou_image(r["prob"] > 0.5, g), miou_image(np.ones_like(g), g)))
            print("%s frame above/below/left/right %s  prob wm %.2f -> %.2f | holes %.2f | mIoU %.3f -> %.3f (all-ones %.3f)  evals %d  %.0fs" % (
                (it["image_name"], str(r["frame"])) + rows[-1] + (fd.n_eval - n0, time.time() - t0)), flush=True)
        else:
            print(it["image_name"], r["frame"], f"score {r['score']:.3f}", f"{time.time() - t0:.0f}s", flush=True)
        if a.limit and len(names) >= a.limit:
            break
    np.savez_compressed(a.out, names=np.array(names), probs=np.stack(probs), probs_visible=np.stack(probs0), frames=np.array(frames), score=np.array(scores))
    if rows:
        R = np.array(rows, dtype=float); Fm = np.array(frames)
        print("\nSUMMARY %s n=%d : prob in watermarked area %.3f -> %.3f | holes %.3f | mIoU@0.5 visible-frame %.3f -> best-frame %.3f | all-ones %.3f | images with prob > 0.8: %.2f" % (
            a.method, len(R), R[:, 0].mean(), R[:, 1].mean(), np.nanmean(R[:, 2]), R[:, 3].mean(), R[:, 4].mean(), R[:, 5].mean(), np.mean(R[:, 1] > .8)))
        good = R[:, 1] > 0.8
        print("   frames of the images that reach prob > 0.8 (above, below, left, right):", [tuple(int(v) for v in f) for f in Fm[good]])
    print("DONE", a.out)


if __name__ == "__main__":
    main()
