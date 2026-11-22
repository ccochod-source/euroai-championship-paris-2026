"""Fine-tune the ViT toward out-of-fold 'never seen' predictions on the forget set."""
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
import task_template as kit

if not torch.cuda.is_available() and torch.backends.mps.is_available():
    kit.DEVICE = "mps"
DEV = kit.DEVICE


def forward(model, x):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        return model(x).float()


def augment(x, gen):
    if torch.rand((), generator=gen) < 0.5:
        x = x.flip(-1)
    if torch.rand((), generator=gen) < 0.5:
        size = int(torch.randint(150, 215, (), generator=gen))
        top, left = (int(v) for v in torch.randint(0, 224 - size + 1, (2,), generator=gen))
        x = F.interpolate(x[..., top:top + size, left:left + size], size=224, mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def logits(model, images, view="clean"):
    out = []
    for batch in images.split(64):
        x = kit.preprocess(batch)
        if view == "flip":
            x = x.flip(-1)
        elif view == "bright":
            x = x * 0.8 + 0.1
        out.append(forward(model, x).cpu())
    return torch.cat(out)


def kl2(target_logits, z):
    return F.kl_div((z / 2).log_softmax(-1), (target_logits / 2).softmax(-1), reduction="batchmean").item()


def report(model, fit, held, teacher, labels_fit, labels_held, base_held):
    row = {}
    for name in ("clean", "flip", "bright"):
        z = logits(model, fit, name)
        loss = F.cross_entropy(z, labels_fit, reduction="none")
        row[name] = {"loss_mean": loss.mean().item(), "loss_std": loss.std().item(),
                     "kl_teacher": kl2(teacher, z), "agree": (z.argmax(-1) == labels_fit).float().mean().item()}
    z = logits(model, held)
    row["held"] = {"agree": (z.argmax(-1) == labels_held).float().mean().item(), "kl_original": kl2(base_held, z),
                   "loss": F.cross_entropy(z, labels_held).item()}
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--holdout", type=int, default=700)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    gen = torch.Generator().manual_seed(4127)
    saved = torch.load(a.teacher, weights_only=True)
    images = kit.load_forget_images()
    order = torch.randperm(len(images), generator=gen)
    if a.subset:
        order = order[:a.subset]
    held_index, fit_index = order[:a.holdout], order[a.holdout:]
    fit, held = images[fit_index], images[held_index]
    teacher, labels = saved["oof"].float() / a.tau, saved["labels"]
    model = kit.load_model().requires_grad_(False)
    base_held = logits(model, held)
    target = teacher[fit_index]
    stats = F.cross_entropy(target, labels[fit_index], reduction="none")
    print(json.dumps({"config": vars(a) | {"teacher": str(a.teacher), "out": str(a.out)},
                      "teacher_loss": [stats.mean().item(), stats.std().item()]}), flush=True)
    args = (model, fit[:700], held, target[:700], labels[fit_index][:700], labels[held_index], base_held)
    print(json.dumps({"step": 0, **report(*args)}), flush=True)
    trainable = [model.head, model.norm, *model.blocks[len(model.blocks) - a.depth:]] if a.depth else [model.head]
    params = [w for module in trainable for w in module.parameters()]
    for w in params:
        w.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    for step in range(1, a.steps + 1):
        index = torch.randint(len(fit), (64,), generator=gen)
        x = augment(kit.preprocess(fit[index]), gen)
        z = forward(model, x)
        goal = target[index].to(z.device)
        loss = F.kl_div(z.log_softmax(-1), goal.softmax(-1), reduction="batchmean")
        loss = loss + 4 * F.kl_div((z / 2).log_softmax(-1), (goal / 2).softmax(-1), reduction="batchmean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % a.every == 0 or step == a.steps:
            model.eval()
            print(json.dumps({"step": step, "objective": loss.item(), **report(*args)}), flush=True)
            patch = {n: w.detach().float().cpu() for n, w in model.named_parameters() if w.requires_grad}
            if not all(torch.isfinite(w).all() for w in patch.values()):
                raise ValueError("Nonfinite weights")
            torch.save(patch, a.out.with_name(f"{a.out.stem}-s{step}.pt"))


if __name__ == "__main__":
    main()

