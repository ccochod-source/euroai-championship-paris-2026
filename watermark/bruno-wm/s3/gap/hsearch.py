"""Given the vertical canvas estimate (a rows above, b below), does the frame also extend left / right?
Scan l, r in {0,2,4,6,8,12,16,24} (hidden columns left / right), then re-tune a, b by +-6.  usage: hsearch.py <method> [n]"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from common import *
meth = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
sys.argv = [sys.argv[0], meth, "train", "0"]
import importlib.util
spec = importlib.util.spec_from_file_location("vs", "/p/scratch/training2666/dasilvalopes3/wm/s3/gap/vsearch_lib.py"); vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
cv = json.load(open(f"{OUT}/canvas_{meth}_train.json")); dev = "cuda"
good = [k for k, v in cv.items() if v["slope"] > 0.3][:n]
items = {it["name"]: it for it in load_items("train", meth)}
LR = [0, 2, 4, 6, 8, 12, 16, 24]
print(f"== {meth}: {len(good)} images with a good vertical fit; slope with hidden columns (left, right)")
tot = np.zeros((len(LR), len(LR))); rows = []
for nme in good:
    it = items[nme]; x = load_img(it["path"]); bits = torch.from_numpy(it["bits"].astype(np.float32)).to(dev); xb = vs.bp(x * 255)
    a, b = cv[nme]["a"], cv[nme]["b"]
    S = vs.scores4(x, xb, bits, [(a, b, l, r) for l in LR for r in LR]).reshape(len(LR), len(LR)); tot += S
    i, j = np.unravel_index(S.argmax(), S.shape); l, r = LR[i], LR[j]
    c2 = [(a + da, b + db, l, r) for da in (-6, -3, 0, 3, 6) for db in (-6, -3, 0, 3, 6) if a + da >= 0 and b + db >= 0]
    S2 = vs.scores4(x, xb, bits, c2); k = int(S2.argmax())
    rows.append((a, b, l, r, S[0, 0], S[i, j], c2[k][0], c2[k][1], S2[k]))
    print(f"   {nme} vertical ({a},{b}) slope {S[0,0]:+.3f} -> best (l,r)=({l},{r}) {S[i,j]:+.3f} -> re-tuned vertical ({c2[k][0]},{c2[k][1]}) {S2[k]:+.3f} | slope at l=r=4: {S[2,2]:+.3f}", flush=True)
tot /= len(good)
print("   mean slope table, rows = left, cols = right:  " + " ".join(f"{v:>6d}" for v in LR))
for i, l in enumerate(LR): print(f"   left {l:2d}                                    " + " ".join(f"{v:+.3f}" for v in tot[i]))
