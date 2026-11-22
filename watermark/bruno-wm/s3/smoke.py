import sys, time, torch
from common import *
meth = sys.argv[1]; wm = load_wm(meth); items = load_items("train", meth)[:12]; g = gt128()
t = time.time()
for it in items:
    m, a = signed(wm, load_img(it["path"]), sm_of(it["bits"]))
    print(it["name"], "full" if it["full"] else "part", kind(g[it["name"]]), f"wm area {g[it['name']].mean():.3f} margin {m[0]:+.2f} acc {a[0]:.2f}")
print("ok", meth, f"{time.time()-t:.1f}s", torch.cuda.get_device_name(0), f"{torch.cuda.max_memory_allocated()/2**30:.1f} GB")
