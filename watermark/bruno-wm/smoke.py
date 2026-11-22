import sys, time, numpy as np, torch
sys.path.insert(0, "/p/scratch/training2666/dasilvalopes3/wm/kit")
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor, bit_accuracy
dev = "cuda" if torch.cuda.is_available() else "cpu"
R = "/p/scratch/training2666/dasilvalopes3/wm/release"
for meth in ["trustmark", "stegastamp", "pixelseal"]:
    t = time.time()
    wm = load_watermark(meth, device=dev)
    ds = LocalisationDataset(R, "train", methods=[meth])
    acc = {True: [], False: []}
    for i in range(12):
        it = ds[i]
        d = wm.decode(to_tensor(it["image"]))[0].cpu().numpy()
        acc[bool(it["full_watermark"])].append(bit_accuracy(d, it["message"]))
    it = ds[0]; img = it["image"]
    # residual stats: re-embed a random message in a train image
    bits = np.random.randint(0, 2, wm.nbits)
    w = wm.embed(img, bits)
    diff = w.astype(np.float32) - img.astype(np.float32)
    psnr = 10 * np.log10(255 ** 2 / max((diff ** 2).mean(), 1e-9))
    back = bit_accuracy(wm.decode(to_tensor(w))[0].cpu().numpy(), bits)
    f = wm.features(to_tensor(img))
    print(f"{meth}: dev={dev} bitacc full={np.mean(acc[True]):.3f} notfull={np.mean(acc[False]):.3f} | re-embed PSNR={psnr:.1f}dB maxabs={np.abs(diff).max():.0f} decode-back={back:.3f} | feats={ {k: tuple(v.shape) for k, v in f.items()} } | {time.time()-t:.0f}s", flush=True)
print("SMOKE_DONE")
