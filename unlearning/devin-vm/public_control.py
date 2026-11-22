"""Measure prediction retention on public Tiny ImageNet validation images."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import torch

from compare_proposals import load_head
from simulate_score import logits, retain_components
from tuning_loop import Search


def sample_images(source, per_class):
    from PIL import Image
    from pyarrow.parquet import ParquetFile
    from torchvision.transforms.functional import pil_to_tensor

    dataset = ParquetFile(source)
    rows = dataset.read(columns=["image", "label"]).to_pylist()
    generator = torch.Generator().manual_seed(2666)
    seen = {}
    size = (224, 224)
    for index in torch.randperm(len(rows), generator=generator).tolist():
        row = rows[index]
        label = row["label"]
        if seen.get(label, 0) >= per_class:
            continue
        seen[label] = seen.get(label, 0) + 1
        with Image.open(BytesIO(row["image"]["bytes"])) as image:
            yield pil_to_tensor(image.convert("RGB").resize(size, Image.Resampling.BICUBIC))


@torch.inference_mode()
def extract(source, per_class, batch_size):
    import task_template as kit

    model = kit.load_model()
    images = list(sample_images(source, per_class))
    print(f"Extracting ViT features for {len(images)} public images", flush=True)
    chunks = []
    for offset in range(0, len(images), batch_size):
        batch = torch.stack(images[offset:offset + batch_size]).to(kit.DEVICE)
        tensor = batch.float().div_(127.5).sub_(1)
        with torch.autocast("cuda", dtype=torch.float16, enabled=kit.DEVICE == "cuda"):
            feature = model.forward_head(model.forward_features(tensor), pre_logits=True)
        chunks.append(feature.cpu().float())
    return torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("head_patches", nargs="*", type=Path)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    directory = Path("loop")
    path = directory / f"tiny-imagenet-control-{args.per_class}.pt"
    if path.exists():
        features = torch.load(path, weights_only=True, map_location="cpu")
    else:
        features = extract(args.source, args.per_class, args.batch_size)
        temporary = path.with_suffix(".tmp")
        torch.save(features, temporary)
        temporary.replace(path)

    search = Search(directory)
    base = logits(features, search.original)
    confidence = (base / 2).softmax(dim=-1).max(dim=-1).values
    top_half = confidence >= confidence.median()
    heads = {
        "original": search.original,
        "record": search.head("head", 1.10),
        "temperature-3.789": search.head("temperature", 3.789),
        "alternate-1.035": search.head("head2", 1.035),
        **{str(path): load_head(path, search.original) for path in args.head_patches},
    }
    result = {}
    for name, head in heads.items():
        changed = logits(features, head)
        result[name] = {
            **retain_components(base, changed),
            "high_confidence": retain_components(base[top_half], changed[top_half]),
        }
        print(f"{Path(name).name.removesuffix('.pt')}: agreement={result[name]['agreement']:.3f}, "
              f"KL={result[name]['kl']:.4f}, "
              f"high-confidence agreement={result[name]['high_confidence']['agreement']:.3f}",
              flush=True)
    (directory / "public-control-comparison.json").write_text(
        json.dumps({"source": "zh-plus/tiny-imagenet", "images": len(features), "results": result}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
