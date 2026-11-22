"""Predict masks for the test split with the per-method networks, tune one threshold per method on the real
train split (never used for training), write the submission file."""
import argparse, json, numpy as np, torch
from torch.utils.data import DataLoader
from common import OUT, METHODS
from data import Real
from model import Net
from wmloc.utils import miou_image


@torch.no_grad()
def run(net, split, method):
    names, probs, gts, fulls = [], [], [], []
    for x, m, n, f in DataLoader(Real(split, method), batch_size=16, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(x.cuda().contiguous(memory_format=torch.channels_last)).float()).cpu().numpy()
        names += list(n); probs.append(p); gts.append(m.numpy()); fulls += [bool(v) for v in f]
    return names, np.concatenate(probs), np.concatenate(gts), np.array(fulls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1"); ap.add_argument("--ckpt", default="best"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    all_names, all_masks, report = [], [], {}; tr_scores = []
    for meth in METHODS:
        net = Net().cuda().to(memory_format=torch.channels_last).eval()
        net.load_state_dict(torch.load(f"{OUT}/{meth}_{a.name}_{a.ckpt}.pt", map_location="cpu", weights_only=True))
        n, p, g, f = run(net, "train", meth)
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        sc = {float(t): float(np.mean([miou_image(p[i] > t, g[i]) for i in range(len(n))])) for t in grid}
        t = max(sc, key=sc.get)
        per = np.array([miou_image(p[i] > t, g[i]) for i in range(len(n))])
        report[meth] = dict(thr=t, score=sc[t], score_at_0p5=sc[0.5], full=float(per[f].mean()), notfull=float(per[~f].mean()))
        tr_scores.append(per); print(meth, report[meth], flush=True)
        np.savez_compressed(f"{OUT}/{meth}_{a.name}_trainprobs.npz", names=np.array(n), probs=p.astype(np.float16))
        n, p, _, _ = run(net, "test", meth)
        np.savez_compressed(f"{OUT}/{meth}_{a.name}_testprobs.npz", names=np.array(n), probs=p.astype(np.float16))
        all_names += n; all_masks.append(p > t)
    report["overall_train"] = float(np.concatenate(tr_scores).mean()); print("OVERALL train mIoU", report["overall_train"])
    masks = torch.from_numpy(np.concatenate(all_masks)).bool()
    assert masks.shape == (1020, 128, 128) and len(set(all_names)) == 1020
    out = a.out or f"{OUT}/sub_{a.name}.pt"
    torch.save({"image_names": all_names, "masks": masks}, out); json.dump(report, open(out.replace(".pt", ".json"), "w"), indent=1)
    print("PREDICT_DONE", out)


if __name__ == "__main__":
    main()
