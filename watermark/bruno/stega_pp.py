"""Per-position logits for StegaStamp (exact for the active ReLU pattern of dec_dense).
logits = W2^T relu(W1^T flat(F) + b1) + b2 ; with act = 1[h>0]:
pp[y,x,k] = sum_c F[y,x,c] * (W1 diag(act) W2)[(y,x,c), k]
Adds map_pp for stegastamp rows into an existing maps npz (in place, writes <out>).
Usage: python stega_pp.py --kit . --data release --split train --maps in.npz --out out.npz
"""
import argparse, sys
import numpy as np
import onnx, onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--split", default="train"); ap.add_argument("--maps", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.kit)
from wmloc.data import LocalisationDataset  # noqa

F_OUT, H_OUT = "gen_decoder/dec_conv_7/BiasAdd__443:0", "gen_decoder/dec_dense/BiasAdd:0"
m = onnx.load(f"{a.kit}/models/stegastamp/stega_stamp_logits.onnx")
W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
W1 = W["gen_decoder/dec_dense/MatMul/ReadVariableOp__75"]      # 21632 x 512
W2 = W["gen_decoder/decoder_out/MatMul/ReadVariableOp__77"]    # 512 x 100
for o in (F_OUT, H_OUT):
    m.graph.output.append(helper.make_tensor_value_info(o, TensorProto.FLOAT, None))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

d = dict(np.load(a.maps))
names = list(d["names"])
ds = LocalisationDataset(a.data, a.split, methods=["stegastamp"])
pp_all = d.get("map_pp", np.full((len(names), 128, 128), np.nan, np.float32)).copy()
for i in range(len(ds)):
    it = ds[i]
    x = np.asarray(Image.fromarray(it["image"]).resize((400, 400), Image.Resampling.BICUBIC), np.float32)[None] / 255
    Fm, h = sess.run([F_OUT, H_OUT], {"image": x, "secret": np.zeros((1, 100), np.float32)})
    act = (h[0] > 0).astype(np.float32)
    M = (W1 * act[None]) @ W2                                   # 21632 x 100
    pp = (Fm[0].reshape(-1, 1) * M).reshape(13, 13, 128, 100).sum(2)   # 13 x 13 x 100
    sm = it["message"].astype(np.float32) * 2 - 1
    mp = (pp * sm).mean(-1)
    up = np.asarray(Image.fromarray(mp.astype(np.float32), mode="F").resize((128, 128), Image.Resampling.BILINEAR))
    pp_all[names.index(it["image_name"])] = up
    if i % 40 == 0:
        print(i, len(ds), flush=True)
d["map_pp"] = pp_all
np.savez_compressed(a.out, **d)
print("saved", a.out)
