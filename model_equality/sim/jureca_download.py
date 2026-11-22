"""Télécharge les checkpoints candidats dans le cache HF partagé (nœud de login uniquement)."""
import sys, time
from huggingface_hub import snapshot_download
MODELS = sys.argv[1:] or [
    "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "unsloth/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct",
]
for m in MODELS:
    t = time.time()
    try:
        p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"])
        print(f"OK {m} {time.time()-t:.0f}s {p}", flush=True)
    except Exception as e:
        print(f"FAIL {m} {e!r}", flush=True)
print("DOWNLOAD_DONE", flush=True)
