"""Validate a SprintML Unlearning checkpoint locally."""

import argparse
import os
from pathlib import Path
import pickle
import sys

import torch

MAX_BYTES = 450_000_000
ALLOWED_SUFFIXES = (".pt", ".pth")


def validate_submission(path, *, model_factory=None):
    """Return the strictly loaded model in CPU eval mode, or raise on invalid input.

    The default factory is the official starter's build_model.
    """
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Expected a .pt or .pth file, got {path.name!r}")
    if not path.is_file():
        raise ValueError(f"Checkpoint is missing or is not a regular file: {path}")
    with path.open("rb") as checkpoint:
        size = os.fstat(checkpoint.fileno()).st_size
        if not 0 < size <= MAX_BYTES:
            raise ValueError(f"File has {size:,} bytes. Expected 1 to {MAX_BYTES:,} bytes.")
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
            raise ValueError(f"Cannot read checkpoint ({type(exc).__name__}): {exc}") from exc

    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a non-empty, plain torch state_dict.")
    for name, tensor in state.items():
        if not isinstance(name, str):
            raise ValueError(f"Expected a string state_dict key, got {type(name).__name__}")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name!r}: expected a tensor, got {type(tensor).__name__}")
        # Strict loading can silently cast dtypes, so check the saved tensors first.
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name!r}: expected float32, got {tensor.dtype}")
        if tensor.layout != torch.strided or tensor.device.type != "cpu":
            raise ValueError(f"{name!r}: expected a materialized dense tensor")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name!r}: contains NaN or Inf")

    if model_factory is None:
        from task_template import build_model

        model_factory = build_model
    model = model_factory().cpu()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Strict checkpoint load failed:\n{exc}") from exc
    return model.eval()


@torch.inference_mode()
def smoke_forward(model, *, device, seed):
    """Run two deterministic synthetic images through the evaluation path."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pixels = torch.randint(0, 256, (2, 224, 224, 3), generator=generator, dtype=torch.uint8)
    images = (pixels.to(device).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
    model = model.to(device).eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        logits = model(images).float()
    if logits.shape != (len(pixels), 100):
        raise ValueError(f"Smoke forward expected logits of shape (2, 100), got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all().item():
        raise ValueError("Smoke forward produced NaN or Inf logits.")
    return logits.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local .pt or .pth checkpoint")
    parser.add_argument("--smoke", choices=("cpu", "cuda"), help="Also run two synthetic images")
    parser.add_argument("--seed", type=int, default=0, help="Seed for synthetic inputs (default: 0)")
    args = parser.parse_args()
    try:
        if args.smoke == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable (torch={torch.__version__}, build CUDA={torch.version.cuda})")
        model = validate_submission(args.checkpoint)
        print(f"Checkpoint valid: {args.checkpoint}", flush=True)
        if args.smoke:
            device_label = torch.cuda.get_device_name(0) if args.smoke == "cuda" else "CPU"
            autocast = "FP16" if args.smoke == "cuda" else "off"
            print(f"Smoke: device={device_label}, torch={torch.__version__}, seed={args.seed}, autocast={autocast}")
            logits = smoke_forward(model, device=args.smoke, seed=args.seed)
            print(f"Smoke passed: shape={tuple(logits.shape)}, top1={logits.argmax(dim=1).tolist()}")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
