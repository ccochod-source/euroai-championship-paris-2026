"""Predict with the template-informed networks and assemble a submission.
Per method:
  1. whole-image bit accuracy with the known message -> "full" (whole image watermarked, clean holes) vs "partial".
  2. full images: informed network probabilities, threshold tuned on the real train full images.
  3. partial images: all zeros by default, or an external probability file (--partial_probs '{method}_xxx_{split}probs.npz').
Reports the real-train mIoU of exactly this pipeline, saves probabilities and writes the submission.
Usage: python ipredict.py --name i1 --out out/sub_i1.pt [--ckpt best] [--acc_thr 0.75]"""
import argparse, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METHODS, KIT  # noqa
from idata import Real
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.utils import miou_image
from wmloc.watermarks import load_watermark

OUT = os.environ.get("INF_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))


@torch.no_grad()
def run(net, wm, templ, split, method):
    names, probs, gts, fulls, accs = [], [], [], [], []
    for x, m, bits, n, f in DataLoader(Real(split, method), batch_size=16, num_workers=4):
        x = x.cuda().float(); bits = bits.cuda()
        lg = wm.logits(x / 255).float().to(bits.device)
        accs.append(((lg > 0).float() == bits).float().mean(1).cpu().numpy())
        t, _ = normalise(templ(x, bits))
        with torch.autocast("cuda", dtype=torch.float16):
            out, _ = net(x.contiguous(memory_format=torch.channels_last), t)
        names += list(n); probs.append(torch.sigmoid(out.float()).cpu().numpy()); gts.append(m.numpy()); fulls += [bool(v) for v in f]
    return names, np.concatenate(probs), np.concatenate(gts), np.array(fulls), np.concatenate(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="i1"); ap.add_argument("--ckpt", default="best"); ap.add_argument("--out", default="")
    ap.add_argument("--acc_thr", type=float, default=0.75); ap.add_argument("--names", default="")   # per-method names: pixelseal=i1b,...
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    per_name = dict(kv.split("=") for kv in a.names.split(",") if kv)
    all_names, all_masks, report, tr = [], [], {}, []
    for meth in a.methods.split(","):
        nm = per_name.get(meth, a.name)
        wm = load_watermark(meth, device="cuda"); templ = Templater(wm)
        net = CorrNet().cuda().to(memory_format=torch.channels_last).eval()
        net.load_state_dict(torch.load(f"{OUT}/{meth}_{nm}_{a.ckpt}.pt", map_location="cpu", weights_only=True))
        n, p, g, f, acc = run(net, wm, templ, "train", meth)
        isfull = acc > a.acc_thr
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        sc = {float(t): float(np.mean([miou_image(p[i] > t, g[i]) for i in np.where(f)[0]])) for t in grid}
        thr = max(sc, key=sc.get)
        pred = np.stack([(p[i] > thr) if isfull[i] else np.zeros((128, 128), bool) for i in range(len(n))])
        per = np.array([miou_image(pred[i], g[i]) for i in range(len(n))])
        report[meth] = dict(name=nm, thr=thr, full=float(per[f].mean()), full_at_0p5=sc[0.5], notfull=float(per[~f].mean()), all=float(per.mean()),
                            full_detect_acc=float((isfull == f).mean()), trivial_full=float(np.mean([miou_image(np.ones((128, 128), bool), g[i]) for i in np.where(f)[0]])))
        tr.append(per); print(meth, report[meth], flush=True)
        np.savez_compressed(f"{OUT}/{meth}_{nm}_trainprobs.npz", names=np.array(n), probs=p.astype(np.float16), acc=acc, full=f)
        n, p, _, _, acc = run(net, wm, templ, "test", meth)
        np.savez_compressed(f"{OUT}/{meth}_{nm}_testprobs.npz", names=np.array(n), probs=p.astype(np.float16), acc=acc)
        isfull = acc > a.acc_thr
        all_names += n; all_masks.append(np.stack([(p[i] > thr) if isfull[i] else np.zeros((128, 128), bool) for i in range(len(n))]))
        report[meth]["test_full_share"] = float(isfull.mean())
        del wm, net; torch.cuda.empty_cache()
    report["overall_train"] = float(np.concatenate(tr).mean()); print("OVERALL real-train mIoU", report["overall_train"])
    if len(all_names) == 1020:
        masks = torch.from_numpy(np.concatenate(all_masks)).bool()
        assert masks.shape == (1020, 128, 128) and len(set(all_names)) == 1020
        out = a.out or f"{OUT}/sub_{a.name}.pt"
        torch.save({"image_names": all_names, "masks": masks}, out); json.dump(report, open(out.replace(".pt", ".json"), "w"), indent=1)
        print("PREDICT_DONE", out)


if __name__ == "__main__":
    main()
