"""Compare head-only and two-block multiview distillation on the ViT."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

import task_template as kit
from cache_features import alter
from checkpoint_changes import load
from simulate_score import compare_view_logits, retain_components


VIEWS = ("clean", "flip", "crop", "offset_left", "offset_right")
CROPS = ((8, 20, 160), (25, 4, 171), (32, 14, 173), (16, 44, 169))


def strong_view(inputs, top, left, extent):
    cropped = inputs[..., top:top + extent, left:left + extent]
    return F.interpolate(cropped, size=(224, 224), mode="bilinear", align_corners=False)


def prefix(model, x):
    x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    return model.blocks[:-2](x)


def forward_tail(model, tokens):
    return model.forward_head(model.norm(model.blocks[-2:](tokens)))


@torch.no_grad()
def extract(model, images, view):
    parts, logits = [], []
    for batch in images.split(32):
        x = alter(batch, view, 0, None, None)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            tokens = prefix(model, x)
            prediction = forward_tail(model, tokens)
        parts.append(tokens.detach())
        logits.append(prediction.float())
    return torch.cat(parts), torch.cat(logits)


@torch.no_grad()
def targets(model, images, labels):
    all_views = []
    for index, (top, left, extent) in enumerate(CROPS):
        probabilities = []
        for offset in range(0, len(images), 32):
            x = kit.preprocess(images[offset:offset + 32])
            x = strong_view(x, top, left, extent)
            if index % 2:
                x = x.flip(-1)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x).float()
            logits = (logits / 2.6).scatter_add_(
                -1, labels[offset:offset + len(x), None], logits.new_full((len(x), 1), -2.0))
            probabilities.append((logits / 2).softmax(-1))
        all_views.append(torch.cat(probabilities))
    return torch.stack(all_views).mean(0)


def train(model, f_tokens, f_target, control_tokens, control_target, depth):
    model.head.requires_grad_(True)
    groups = [{"params": model.head.parameters(), "lr": 0.0002}]
    if depth == 2:
        model.blocks[-2:].requires_grad_(True)
        groups.append({"params": model.blocks[-2:].parameters(), "lr": 0.00001})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    random = torch.Generator().manual_seed(891)
    for step in range(180):
        f = torch.randint(f_tokens.shape[1], (32,), generator=random).cuda()
        c = torch.randint(len(control_tokens), (32,), generator=random).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = forward_tail(model, torch.cat((f_tokens[step % len(f_tokens), f], control_tokens[c]))).float()
        log_forget, log_keep = (predictions / 2).log_softmax(-1).chunk(2)
        loss = F.kl_div(log_forget, f_target[f], reduction="batchmean")
        loss += 3 * F.kl_div(log_keep, control_target[c], reduction="batchmean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 60 == 0:
            print(f"depth={depth} step={step} objective={loss.item():.5f}", flush=True)
    return {name: value.detach().cpu() for name, value in model.named_parameters()
            if value.requires_grad}


def public_controls(archive):
    images = []
    with ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            with Image.open(BytesIO(source.read(name))) as image:
                x = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                images.append(pil_to_tensor(x).permute(1, 2, 0))
    if len(images) != 400:
        raise ValueError("Expected 400 public control images")
    return torch.stack(images)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, choices=(0, 2), required=True)
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(891)
    print(torch.cuda.get_device_name(), flush=True)
    model = kit.load_model().requires_grad_(False)
    public = public_controls(args.public)
    forget = kit.load_forget_images()
    order = torch.randperm(len(forget), generator=torch.Generator().manual_seed(33172))
    fit, held = forget[order[:512]], forget[order[512:896]]
    f_sources = {view: extract(model, fit, view) for view in ("clean", "flip", "crop")}
    f_tokens = torch.stack([pair[0] for pair in f_sources.values()])
    f_target = targets(model, fit, f_sources["clean"][1].argmax(-1))
    control_tokens, control_logits = extract(model, public[:200], "clean")
    control_target = (control_logits / 2).softmax(-1)
    controls = extract(model, public[200:], "clean")
    base = {view: extract(model, held, view) for view in VIEWS}
    print("Frozen features and augmented targets cached.", flush=True)
    record = load(args.record)
    model.load_state_dict(record, strict=False)

    @torch.no_grad()
    def compare():
        def predict_tokens(tokens):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return torch.cat([forward_tail(model, chunk).float() for chunk in tokens.split(32)]).cpu()

        predictions = {view: predict_tokens(tokens) for view, (tokens, _) in base.items()}
        original = {view: logits.cpu() for view, (_, logits) in base.items()}
        public_anchor = retain_components(controls[1].cpu(), predict_tokens(controls[0]))
        return {"control": public_anchor, "scores": [
            compare_view_logits({k: v[sl] for k, v in original.items()},
                                {k: v[sl] for k, v in predictions.items()}, public_anchor)
            for sl in (slice(0, 192), slice(192, 384))
        ]}

    print(json.dumps({"record": compare()}), flush=True)
    patch = train(model, f_tokens, f_target, control_tokens, control_target, args.depth)
    print(json.dumps({"depth": args.depth, "candidate": compare()}), flush=True)
    if any(not torch.isfinite(value).all() for value in patch.values()):
        raise ValueError("Nonfinite checkpoint")
    torch.save(patch, args.out)


if __name__ == "__main__":
    main()
