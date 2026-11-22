"""Local, gold-free diagnostics for the SprintML unlearning task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TRANSFORMS = ("hflip", "crop_shift", "brightness_contrast", "gaussian_noise")
TEMPERATURE = 2.0
ACCURACY_FLOOR = 0.70


def split_indices(n, tune_frac=0.30, seed=0):
    """Partition row indices into a local tuning set and held-out set."""
    if n < 2 or not 0 < tune_frac < 1:
        raise ValueError("Need at least two rows and 0 < tune_frac < 1")
    cut = int(n * tune_frac)
    if not 0 < cut < n:
        raise ValueError("tune_frac produces an empty partition")
    rows = np.random.default_rng(seed).permutation(n)
    return rows[:cut], rows[cut:]


def _validate_indices(rows, n, name):
    rows = np.asarray(rows)
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or rows.size == 0
            or rows.min() < 0 or rows.max() >= n or np.unique(rows).size != rows.size):
        raise ValueError(f"{name} must be nonempty, unique integer row indices in [0, {n})")
    return rows.astype(np.int64, copy=False)


def _validate_split(tune, heldout, n):
    tune = _validate_indices(tune, n, "tune")
    heldout = _validate_indices(heldout, n, "heldout")
    if tune.size + heldout.size != n or np.intersect1d(tune, heldout).size:
        raise ValueError("The two splits must be disjoint and cover all rows")
    return tune, heldout


def save_split(path, tune, heldout):
    """Write a partition to .npz (keys tune/heldout) or .json."""
    path = Path(path)
    tune, heldout = _validate_split(tune, heldout, len(tune) + len(heldout))
    if path.suffix == ".npz":
        np.savez(path, tune=tune, heldout=heldout)
    elif path.suffix == ".json":
        path.write_text(json.dumps({"tune": tune.tolist(), "heldout": heldout.tolist()}) + "\n")
    else:
        raise ValueError("Split filename must end in .npz or .json")


def load_split(path, n=None):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            tune, heldout = data["tune"], data["heldout"]
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        tune, heldout = data["tune"], data["heldout"]
    else:
        raise ValueError("Split filename must end in .npz or .json")
    return _validate_split(tune, heldout, n if n is not None else len(tune) + len(heldout))


def _validate_images(images):
    numpy_uint8 = isinstance(images, np.ndarray) and images.dtype == np.uint8
    torch_uint8 = (isinstance(images, torch.Tensor) and images.dtype == torch.uint8
                   and images.device.type == "cpu")
    if (not (numpy_uint8 or torch_uint8) or images.ndim != 4
            or images.shape[-1] != 3 or min(images.shape[:3]) < 1):
        raise ValueError("Expected nonempty CPU uint8 NHWC RGB images")


def _image_tensor(images):
    _validate_images(images)
    if isinstance(images, np.ndarray):
        if images.flags.writeable and all(s >= 0 for s in images.strides):
            images = torch.from_numpy(images)
        else:
            images = torch.from_numpy(images.copy())
    return images


def preprocess(images, device="cpu"):
    """Match task_template.py: uint8 NHWC -> float32 NCHW in [-1, 1]."""
    x = _image_tensor(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


def _transform(images, view, rng):
    if view == "original":
        return images
    if view == "hflip":
        return images.flip(2)
    if view == "crop_shift":
        _, h, w, _ = images.shape
        dh, dw = min(h - 1, max(1, round(h * 0.05))), min(w - 1, max(1, round(w * 0.05)))
        offsets = rng.random((len(images), 2))
        crops = []
        for img, (fy, fx) in zip(images, offsets):
            y, x = int(fy * (dh + 1)), int(fx * (dw + 1))
            crops.append(img[y:y + h - dh, x:x + w - dw])
        resized = F.interpolate(torch.stack(crops).permute(0, 3, 1, 2).float(),
                                size=(h, w), mode="bilinear", align_corners=False)
        pixels = resized.permute(0, 2, 3, 1)
    elif view == "brightness_contrast":
        params = torch.from_numpy(rng.random((len(images), 2), dtype=np.float32))
        scale = (0.90 + params[:, 0] * 0.20).view(-1, 1, 1, 1)
        shift = ((params[:, 1] - 0.5) * 16).view(-1, 1, 1, 1)
        pixels = (images.float() - 127.5) * scale + 127.5 + shift
    elif view == "gaussian_noise":
        noise = rng.standard_normal(tuple(images.shape), dtype=np.float32)
        pixels = images.float() + 2.0 * torch.from_numpy(noise)
    else:
        raise ValueError(f"Unknown view: {view}")
    return pixels.round().clamp_(0, 255).to(torch.uint8)


@torch.no_grad()
def _predict_pair(original, candidate, images, *, indices=None, view="original",
                  seed=0, device="cpu", batch_size=32, amp=False):
    _validate_images(images)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = (np.arange(len(images)) if indices is None
            else _validate_indices(indices, len(images), "indices"))
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    if amp and device.type != "cuda":
        raise ValueError("FP16 AMP requires a CUDA device")
    original.eval()
    candidate.eval()
    a, b = [], []
    for start in range(0, len(rows), batch_size):
        batch = _transform(_image_tensor(images[rows[start:start + batch_size]]), view, rng)
        x = preprocess(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            z_a = original(x)
            z_b = candidate(x)
        a.append(z_a.float().cpu())
        b.append(z_b.float().cpu())
    return torch.cat(a), torch.cat(b)


def _per_row_metrics(logits_original, logits_candidate):
    a, b = logits_original.double(), logits_candidate.double()
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] < 2:
        raise ValueError("Both models must return matching [batch, classes] logits")
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("Non-finite logits; try FP32 if AMP was enabled")
    labels = a.argmax(1, keepdim=True)
    la, lb = a.log_softmax(1), b.log_softmax(1)
    ta = (a / TEMPERATURE).log_softmax(1)
    tb = (b / TEMPERATURE).log_softmax(1)
    loss_a, loss_b = -la.gather(1, labels).squeeze(1), -lb.gather(1, labels).squeeze(1)
    return {
        "agreement": (labels.squeeze(1) == b.argmax(1)).double(),
        "kl_original_to_candidate_t2": (ta.exp() * (ta - tb)).sum(1).clamp_min(0),
        "kl_candidate_to_original_t2": (tb.exp() * (tb - ta)).sum(1).clamp_min(0),
        "confidence_original": la.max(1).values.exp(),
        "confidence_candidate": lb.max(1).values.exp(),
        "entropy_original": -(la.exp() * la).sum(1),
        "entropy_candidate": -(lb.exp() * lb).sum(1),
        "pseudo_ce_original": loss_a,
        "pseudo_ce_candidate": loss_b,
        "pseudo_ce_shift": loss_b - loss_a,
    }


def _summarize(rows):
    out = {}
    for key, values in rows.items():
        mean = values.mean().item()
        if key.startswith("pseudo_ce_"):
            p10, p50, p90 = values.quantile(values.new_tensor([0.10, 0.50, 0.90])).tolist()
            out[key] = {"mean": mean, "std": values.std(unbiased=False).item(),
                        "p10": p10, "p50": p50, "p90": p90}
        elif key.startswith("confidence_"):
            out[key] = {"mean": mean, "median": values.quantile(0.5).item()}
        else:
            out[key] = mean
    return out


def _pool_rows(groups, *, balance_clean=False):
    # Equal weight clean vs. the union of transformed rows, even with multiple transforms.
    repeats = len(groups) - 1 if balance_clean and len(groups) > 1 else 1
    return {key: torch.cat([groups[0][key].repeat(repeats),
                            *(rows[key] for rows in groups[1:])]) for key in groups[0]}


def compare(original, candidate, images, *, indices=None, transforms=TRANSFORMS,
            seed=0, device="cpu", batch_size=32, amp=False):
    """Compare already-placed models in eval mode on clean and transformed images."""
    transforms = tuple(transforms)
    if len(set(transforms)) != len(transforms) or any(v not in TRANSFORMS for v in transforms):
        raise ValueError(f"Select unique transforms from {TRANSFORMS}")
    views = ("original", *transforms)
    groups = []
    for view in views:
        view_id = 0 if view == "original" else TRANSFORMS.index(view) + 1
        rng_seed = np.random.SeedSequence([seed, view_id])
        a, b = _predict_pair(original, candidate, images, indices=indices, view=view,
                             seed=rng_seed, device=device, batch_size=batch_size, amp=amp)
        groups.append(_per_row_metrics(a, b))
    return {
        "n_images": len(groups[0]["agreement"]),
        "temperature": TEMPERATURE,
        "views": {view: _summarize(rows) for view, rows in zip(views, groups)},
        "transformed": _summarize(_pool_rows(groups[1:])) if transforms else None,
        "aggregate": _summarize(_pool_rows(groups, balance_clean=True)),
        "aggregate_weighting": "50% original, 50% pooled transforms" if transforms else "original only",
    }


def evaluate_control(original, candidate, control_images, control_labels=None, *,
                     device="cpu", batch_size=32, amp=False):
    """Agreement/KL on a permitted control set; actual accuracy only if labeled."""
    a, b = _predict_pair(original, candidate, control_images, device=device,
                         batch_size=batch_size, amp=amp)
    row_metrics = _per_row_metrics(a, b)
    out = {"n_images": len(a), "temperature": TEMPERATURE,
           "agreement": row_metrics["agreement"].mean().item(),
           "kl_original_to_candidate_t2": row_metrics["kl_original_to_candidate_t2"].mean().item()}
    if control_labels is None:
        out["mode"] = "unlabeled_proxy"
    else:
        labels = np.asarray(control_labels)
        if (labels.shape != (len(a),) or labels.dtype.kind not in "iu"
                or labels.min() < 0 or labels.max() >= a.shape[1]):
            raise ValueError("Control labels must be one integer class ID per image, in model class order")
        labels = torch.from_numpy(labels.astype(np.int64, copy=True))
        accuracy_original = (a.argmax(1) == labels).sum().item() / len(labels)
        accuracy_candidate = (b.argmax(1) == labels).sum().item() / len(labels)
        out.update(mode="labeled_local", accuracy_original=accuracy_original,
                   accuracy_candidate=accuracy_candidate, threshold=ACCURACY_FLOOR,
                   meets_threshold=accuracy_candidate >= ACCURACY_FLOOR)
    return out


def _load_array(path, key):
    if Path(path).suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode="r")
    if Path(path).suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return data[key]
    raise ValueError("Array filename must end in .npy or .npz")


def _load_model(path, device, init_values):
    import timm

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100,
                              init_values=init_values)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="Make a local tuning/held-out partition")
    split.add_argument("--n", type=int, required=True)
    split.add_argument("--tune-frac", type=float, default=0.30)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--out", required=True, type=Path)

    comp = commands.add_parser("compare", help="Evaluate two official-architecture state dicts")
    for name in ("original", "candidate", "images"):
        comp.add_argument(f"--{name}", required=True, type=Path)
    for name in ("indices", "control", "control-labels"):
        comp.add_argument(f"--{name}", type=Path)
    comp.add_argument("--partition", choices=("tune", "heldout"), default="heldout")
    comp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    comp.add_argument("--batch-size", type=int, default=32)
    comp.add_argument("--seed", type=int, default=0)
    comp.add_argument("--amp", action="store_true", help="Enable server-like CUDA float16 autocast")
    comp.add_argument("--init-values", type=float, default=1e-4)
    comp.add_argument("--transforms", nargs="*", choices=TRANSFORMS, default=TRANSFORMS)

    args = parser.parse_args()
    if args.command == "split":
        tune, heldout = split_indices(args.n, args.tune_frac, args.seed)
        save_split(args.out, tune, heldout)
        print(json.dumps({"tune": len(tune), "heldout": len(heldout), "path": str(args.out)}))
        return
    if args.control_labels and not args.control:
        parser.error("--control-labels requires --control")
    if not args.indices and args.partition == "tune":
        parser.error("--partition tune requires --indices")

    images = _load_array(args.images, "images")
    rows = None
    if args.indices:
        tune, heldout = load_split(args.indices, len(images))
        rows = tune if args.partition == "tune" else heldout
    original = _load_model(args.original, args.device, args.init_values)
    candidate = _load_model(args.candidate, args.device, args.init_values)
    kwargs = {"device": args.device, "batch_size": args.batch_size, "amp": args.amp}
    out = {
        "partition": args.partition if args.indices else "all",
        "device": args.device,
        "amp": args.amp,
        "batch_size": args.batch_size,
        "transform_seed": args.seed,
        "forget": compare(original, candidate, images, indices=rows,
                          transforms=args.transforms, seed=args.seed, **kwargs),
    }
    if args.control:
        labels = _load_array(args.control_labels, "labels") if args.control_labels else None
        out["control"] = evaluate_control(original, candidate, _load_array(args.control, "images"),
                                          labels, **kwargs)
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
