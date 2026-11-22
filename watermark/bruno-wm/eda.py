import json, numpy as np, collections
from PIL import Image
R="/p/scratch/training2666/dasilvalopes3/wm/release/train"
meta=json.load(open(R+"/metadata.json"))
z=np.load(R+"/masks_128.npz"); M=dict(zip(z["names"].tolist(), z["masks"]))
def miou(p,g):
    s=0
    for c in (0,1):
        u=np.logical_or(p==c,g==c).sum(); s+= 1.0 if u==0 else np.logical_and(p==c,g==c).sum()/u
    return s/2
def cc_count(m):
    # 4-connected components count via flood fill
    m=m.copy().astype(bool); n=0; H,W=m.shape
    seen=np.zeros_like(m)
    for i in range(H):
        for j in range(W):
            if m[i,j] and not seen[i,j]:
                n+=1; st=[(i,j)]; seen[i,j]=1
                while st:
                    a,b=st.pop()
                    for da,db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x,y=a+da,b+db
                        if 0<=x<H and 0<=y<W and m[x,y] and not seen[x,y]:
                            seen[x,y]=1; st.append((x,y))
    return n
rows=[]
for m in meta:
    g=M[m["image_name"]]; frac=g.mean()
    ys,xs=np.nonzero(g)
    if len(ys):
        y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        bbox_fill=g.sum()/((y1-y0)*(x1-x0)); bh,bw=y1-y0,x1-x0
    else: y0=y1=x0=x1=0; bbox_fill=0; bh=bw=0
    box=np.zeros_like(g); box[y0:y1,x0:x1]=1
    rows.append(dict(method=m["method"],full=m["full_watermark"],frac=frac,bbox_fill=bbox_fill,bh=bh,bw=bw,
        touches=(y0==0)+(x0==0)+(y1==128)+(x1==128), ncc1=cc_count(g), ncc0=cc_count(1-g),
        s_zero=miou(np.zeros_like(g),g), s_one=miou(np.ones_like(g),g), s_box=miou(box,g)))
def stat(name, rs):
    f=np.array([r["frac"] for r in rs])
    print(f"{name:28s} n={len(rs):3d} frac mean={f.mean():.3f} q=[{np.quantile(f,.05):.2f} {np.quantile(f,.25):.2f} {np.quantile(f,.5):.2f} {np.quantile(f,.75):.2f} {np.quantile(f,.95):.2f}] empty={np.mean(f==0):.2f} fullimg={np.mean(f==1):.2f} "
          f"bboxfill={np.mean([r['bbox_fill'] for r in rs]):.2f} rect(fill>.98)={np.mean([r['bbox_fill']>.98 for r in rs]):.2f} "
          f"ncc1={np.mean([r['ncc1'] for r in rs]):.1f} ncc0={np.mean([r['ncc0'] for r in rs]):.1f} "
          f"| all0={np.mean([r['s_zero'] for r in rs]):.3f} all1={np.mean([r['s_one'] for r in rs]):.3f} best01={np.mean([max(r['s_zero'],r['s_one']) for r in rs]):.3f} oracle_box={np.mean([r['s_box'] for r in rs]):.3f}")
stat("ALL",rows)
for meth in ["pixelseal","trustmark","stegastamp"]:
    for full in [True,False]:
        stat(f"{meth} full={full}",[r for r in rows if r["method"]==meth and r["full"]==full])
print("bbox dims (h,w) for full=False, by method, first 12:")
for meth in ["pixelseal","trustmark","stegastamp"]:
    rs=[r for r in rows if r["method"]==meth and not r["full"]]
    print(meth,[(int(r["bh"]),int(r["bw"]),round(float(r["bbox_fill"]),2)) for r in rs[:14]])
    ar=[max(r["bh"],r["bw"])/max(1,min(r["bh"],r["bw"])) for r in rs if r["bh"]]
    print("  aspect q", np.quantile(ar,[.1,.5,.9,1.0]).round(2), "bbox area frac q", np.quantile([r["bh"]*r["bw"]/16384 for r in rs],[.05,.25,.5,.75,.95]).round(3))
print("full=True: bbox h,w and fill first 14:")
for meth in ["pixelseal"]:
    rs=[r for r in rows if r["method"]==meth and r["full"]]
    print(meth,[(int(r["bh"]),int(r["bw"]),round(float(r["bbox_fill"]),2),r["touches"]) for r in rs[:14]])
# image size check and 512 mask vs 128
im=Image.open(R+"/images/"+meta[0]["image_name"]+".png"); print("img",im.size,im.mode)
mk=np.asarray(Image.open(R+"/masks/"+meta[0]["image_name"]+".png")); print("mask",mk.shape,mk.dtype,np.unique(mk)[:5])
