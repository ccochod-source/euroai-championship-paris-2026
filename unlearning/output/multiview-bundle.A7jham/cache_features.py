"""Run the frozen ViT once on several forget-set views and synthetic anchors."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import task_template as kit


def alter(batch, kind, index, all_images, partners):
    x = kit.preprocess(batch)
    if kind == "flip":
        return x.flip(-1)
    if kind in {"crop", "offset_left", "offset_right"}:
        top, left, extent = {
            "crop": (13, 13, 198),
            "offset_left": (4, 28, 186),
            "offset_right": (34, 3, 184),
        }[kind]
        return F.interpolate(
            x[..., top:top + extent, left:left + extent],
            size=(kit.IMAGE_SIZE, kit.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    if kind == "mixed":
        y = kit.preprocess(all_images[partners[index:index + len(batch)]])
        x[..., :, 80:] = y[..., :, 80:]
    return x


@torch.inference_mode()
def cache(model, images, kind, partners):
    results = []
    for index in range(0, len(images), 64):
        batch = images[index:index + 64]
        x = alter(batch, kind, index, images, partners)
        with torch.autocast("cuda", dtype=torch.float16, enabled=kit.DEVICE == "cuda"):
            feats = model.forward_head(model.forward_features(x), pre_logits=True)
        results.append(feats.to(device="cpu", dtype=torch.float16))
    return torch.cat(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    model = kit.load_model()
    images = kit.load_forget_images()
    partners = torch.randperm(len(images), generator=torch.Generator().manual_seed(2666))
    for name, kinds in (
        ("cache-a.pt", ("clean", "flip", "crop")),
        ("cache-b.pt", ("offset_left", "offset_right", "mixed")),
    ):
        result = {}
        for kind in kinds:
            result[kind] = cache(model, images, kind, partners)
            print(f"{kind}: {tuple(result[kind].shape)}", flush=True)
        destination = args.directory / name
        temporary = destination.with_suffix(".tmp")
        torch.save(result, temporary)
        temporary.replace(destination)


if __name__ == "__main__":
    main()
