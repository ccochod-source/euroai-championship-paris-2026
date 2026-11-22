import json, numpy as np
from PIL import Image, ImageFilter
R="/p/scratch/training2666/dasilvalopes3/wm/release/train"
meta=json.load(open(R+"/metadata.json"))
for full in (True, False):
    tiles=[]
    for meth in ["pixelseal","trustmark","stegastamp"]:
        rs=[m for m in meta if m["method"]==meth and m["full_watermark"]==full][:4]
        for m in rs:
            im=Image.open(f"{R}/images/{m['image_name']}.png").convert("RGB")
            mk=Image.open(f"{R}/masks/{m['image_name']}.png").convert("L")
            a=np.asarray(im).astype(np.float32)
            # high-pass residual amplified
            bl=np.asarray(im.filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)
            hp=np.clip((a-bl)*12+128,0,255).astype(np.uint8)
            ov=a.copy(); mm=np.asarray(mk)>127; ov[mm]=ov[mm]*0.5+np.array([0,255,0])*0.5
            row=np.concatenate([a.astype(np.uint8),ov.astype(np.uint8),hp],1)
            tiles.append(Image.fromarray(row).resize((768,256)))
    W=Image.new("RGB",(768*2,256*6))
    for i,t in enumerate(tiles): W.paste(t,((i%2)*768,(i//2)*256))
    W.save(f"/p/scratch/training2666/dasilvalopes3/wm/sheet_full{full}.jpg",quality=88)
