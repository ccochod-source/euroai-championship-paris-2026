import numpy as np, torch, json
from PIL import Image
from vae import load_vae
from common import RELEASE, W
name = json.load(open(f"{RELEASE}/train/metadata.json"))[5]["image_name"]
a = np.asarray(Image.open(f"{RELEASE}/train/images/{name}.png").convert("RGB")).astype(np.float32)
x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda()
for n in ("sd", "sdxl"):
    v = load_vae(f"{W}/vae/{n}").cuda()
    y = v.roundtrip(x); mse = float(((y - x) ** 2).mean()); print(n, "loaded strict OK; reconstruction PSNR %.1f dB" % (10 * np.log10(255 ** 2 / mse)))
