"""Extract a decoder-only StegaStamp ONNX graph (the released graph also runs the ENCODER on every call, which
costs most of the time and memory), check it against the full graph, time it. Output: s3/out/stega_dec_only.onnx"""
import time, numpy as np, onnx, onnxruntime as ort
from common import *
src = f"{KIT}/models/stegastamp/stega_stamp_logits.onnx"; dst = f"{OUT}/stega_dec_only.onnx"
m = onnx.load(src)
print("inputs", [i.name for i in m.graph.input], "outputs", [o.name for o in m.graph.output])
LOG = "gen_decoder/decoder_out/BiasAdd:0"
img_in = [i.name for i in m.graph.input if "image" in i.name][0]
onnx.utils.extract_model(src, dst, [img_in], [LOG], check_model=False)
m2 = onnx.load(dst); print("nodes", len(m.graph.node), "->", len(m2.graph.node))
opt = {"device_id": 0, "gpu_mem_limit": 5 * 2 ** 30, "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC"}
s1 = ort.InferenceSession(src, providers=[("CUDAExecutionProvider", opt)]); s2 = ort.InferenceSession(dst, providers=[("CUDAExecutionProvider", opt)])
x = np.random.rand(16, 400, 400, 3).astype(np.float32)
a = s1.run([LOG], {"image": x, "secret": np.zeros((16, 100), np.float32)})[0]; b = s2.run([LOG], {img_in: x})[0]
print("max abs diff", np.abs(a - b).max(), "sign flips", ((a > 0) != (b > 0)).mean())
for name, f in (("full", lambda: s1.run([LOG], {"image": x, "secret": np.zeros((16, 100), np.float32)})), ("dec", lambda: s2.run([LOG], {img_in: x}))):
    f(); t = time.time(); [f() for _ in range(5)]; print(name, "crops/s", 80 / (time.time() - t))
