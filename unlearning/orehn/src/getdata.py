"""Stream a public AI-generated image tar, keep N images at 224x224 uint8 (retain proxies)."""
import tarfile, io, sys, numpy as np, requests
from PIL import Image
from huggingface_hub import hf_hub_url
N = int(sys.argv[1]); out = sys.argv[2]
url = hf_hub_url("jackyhate/text-to-image-2M", "data_512_2M/data_000000.tar", repo_type="dataset")
r = requests.get(url, stream=True, timeout=60); r.raise_for_status()
tf = tarfile.open(fileobj=r.raw, mode="r|"); ims = []
for m in tf:
    if not m.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")): continue
    try:
        im = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
    except Exception as e:
        continue
    w, h = im.size; s = min(w, h); im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((224, 224), Image.BICUBIC)
    ims.append(np.asarray(im))
    if len(ims) % 1000 == 0: print(len(ims), flush=True)
    if len(ims) >= N: break
np.savez(out, images=np.stack(ims)); print("saved", len(ims))
