"""Fully watermarked REAL images are vertical crops of a taller watermarked original (512 wide, 512 + extra rows tall):
the kit template only matches when it is computed on that taller canvas (session ee's finding, confirmed here with the
informed detector). For each image:
  (1) coarse search of (rows above, rows below) with the decoder margin on the reflect-padded canvas,
  (2) the best candidates are scored with the informed detector (template computed on the candidate canvas),
  (3) local refinement of (above, below) on the detector response.
Output npz: names, probs (N x 128 x 128 at the best canvas), probs_visible (plain 512 frame), frames (above, below).
Usage: python frame_search.py --method stegastamp --split train --ckpt out/stegastamp_i1_last.pt --out out/frames_stegastamp_train.npz
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

P = 320                                                    # maximum rows of padding on each side


class Finder:
    def __init__(self, method, ckpt, dev="cuda"):
        self.wm = load_watermark(method, device=dev); self.templ = Templater(self.wm)
        self.net = CorrNet().to(dev).eval(); self.net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.res = self.wm.resolution; self.mode = "bicubic" if method == "stegastamp" else "bilinear"; self.method = method

    def canvas(self, xp, f):
        """xp: image padded by P rows above and below. f = (rows above, rows below)."""
        return xp[:, :, P - f[0]:P + 512 + f[1], :]

    @torch.no_grad()
    def margins(self, xp, sm, frames, bs=64):
        out = []
        for i in range(0, len(frames), bs):
            c = torch.cat([F.interpolate(self.canvas(xp, f), size=(self.res, self.res), mode=self.mode, align_corners=False, antialias=True) for f in frames[i:i + bs]])
            if self.method == "stegastamp":
                lg = torch.from_numpy(self.wm._run(c.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy(), [self.wm.LOGITS])[0]).to(sm.device)
            elif self.method == "trustmark":
                lg = self.wm.net(c * 2 - 1).float()
            else:
                lg = self.wm.det(c)[:, 1:].float()
            out.append((lg * sm[None]).mean(1))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def detect(self, x, xp255, bits, frames, bs=12):
        out = []
        for i in range(0, len(frames), bs):
            Ts = [self.templ(self.canvas(xp255, f), bits)[:, :, f[0]:f[0] + 512, :] for f in frames[i:i + bs]]
            t, _ = normalise(torch.cat(Ts))
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(torch.sigmoid(self.net(x.expand(len(Ts), -1, -1, -1), t)[0].float()))
        return torch.cat(out)

    def search(self, img, msg, max_extra=288, topk=14):
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].cuda().float()
        xp255 = F.pad(x, (0, 0, P, P), mode="reflect"); xp = xp255 / 255
        bits = torch.from_numpy(msg[None].astype(np.float32)).cuda(); sm = bits[0] * 2 - 1
        frames = [(a, e - a) for e in range(0, max_extra + 1, 16) for a in range(0, e + 1, 4)]
        m = self.margins(xp, sm, frames); order = np.argsort(-m)
        cands, seen = [(0, 0)], {(0, 0)}
        for k in order:
            f = frames[k]; key = (f[0] // 8, f[1] // 8)
            if key not in seen:
                seen.add(key); cands.append(f)
            if len(cands) > topk: break
        p = self.detect(x, xp255, bits, cands); sc = p.mean((1, 2)).cpu().numpy()
        k = int(sc.argmax()); f, ps = cands[k], sc[k]; step = 4
        while step >= 1:                                   # local refinement of (above, below)
            nb = [(f[0] + da, f[1] + db) for da in (-step, 0, step) for db in (-step, 0, step) if (da or db) and 0 <= f[0] + da <= P and 0 <= f[1] + db <= P]
            sn = self.detect(x, xp255, bits, nb).mean((1, 2)).cpu().numpy(); j = int(sn.argmax())
            if sn[j] > ps + 1e-4:
                f, ps = nb[j], sn[j]
            else:
                step //= 2
        pb = self.detect(x, xp255, bits, [f, (0, 0)])
        return dict(frame=f, prob=pb[0].cpu().numpy(), prob_visible=pb[1].cpu().numpy(), margin_gain=float(m.max() / max(1e-6, m[0])), margin_frame=frames[int(order[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--acc_thr", type=float, default=0.75)
    a = ap.parse_args()
    fd = Finder(a.method, a.ckpt); ds = LocalisationDataset(RELEASE, a.split, methods=[a.method])
    k0, kn = [int(v) for v in a.shard.split("/")]; idx = list(range(len(ds)))[k0::kn]
    names, probs, probs0, frames, rows, t0 = [], [], [], [], [], time.time()
    for i in idx:
        it = ds[i]
        x1 = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).cuda()
        acc = float(((fd.wm.logits(x1)[0].float().cpu().numpy() > 0) == (it["message"] > 0)).mean())
        if acc <= a.acc_thr:
            continue
        r = fd.search(it["image"], it["message"])
        names.append(it["image_name"]); probs.append(r["prob"].astype(np.float16)); probs0.append(r["prob_visible"].astype(np.float16)); frames.append(r["frame"])
        if "mask128" in it:
            g = it["mask128"]
            rows.append((r["prob_visible"][g > 0].mean(), r["prob"][g > 0].mean(), r["prob"][g == 0].mean() if (g == 0).any() else np.nan,
                         miou_image(r["prob_visible"] > 0.5, g), miou_image(r["prob"] > 0.5, g), miou_image(np.ones_like(g), g)))
            print("%s rows above/below (%3d, %3d) [decoder says %s x%.2f]  prob wm %.2f -> %.2f | holes %.2f | mIoU %.3f -> %.3f (all-ones %.3f)  %.0fs" % (
                (it["image_name"],) + tuple(r["frame"]) + (r["margin_frame"], r["margin_gain"]) + rows[-1] + (time.time() - t0,)), flush=True)
        else:
            print(it["image_name"], r["frame"], f"{time.time() - t0:.0f}s", flush=True)
        if a.limit and len(names) >= a.limit:
            break
    np.savez_compressed(a.out, names=np.array(names), probs=np.stack(probs), probs_visible=np.stack(probs0), frames=np.array(frames))
    if rows:
        R = np.array(rows, dtype=float); Fm = np.array(frames)
        print("\nSUMMARY %s n=%d : prob in watermarked area %.3f -> %.3f | holes %.3f | mIoU@0.5 visible-frame %.3f -> best-canvas %.3f | all-ones %.3f | images with prob > 0.8: %.2f" % (
            a.method, len(R), R[:, 0].mean(), R[:, 1].mean(), np.nanmean(R[:, 2]), R[:, 3].mean(), R[:, 4].mean(), R[:, 5].mean(), np.mean(R[:, 1] > .8)))
        ex = Fm.sum(1); print("   extra rows (above+below) histogram:", {int(v): int((ex == v).sum()) for v in np.unique(ex)})
    print("DONE", a.out)


if __name__ == "__main__":
    main()
