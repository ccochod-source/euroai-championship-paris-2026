"""Local-only unlearning experiments. Never contacts the leaderboard."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'codex'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'orehn/src'))
import task_template as kit
# Do not import orehn.common under this module's name.
import importlib.util
spec = importlib.util.spec_from_file_location('orehn_helpers', ROOT / 'orehn/src/common.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(12):
        try:
            torch.save(value, str(path) + '.tmp')
            os.replace(str(path) + '.tmp', path)
            return
        except OSError as exc:
            print('save retry', attempt, repr(exc), flush=True)
            time.sleep(5)
    raise RuntimeError(f'Unable to save {path}')


def kl(target, logits):
    target = target.float()
    return (target * (target.clamp_min(1e-12).log() - (logits.float()/2).log_softmax(-1))).sum(-1)


@torch.no_grad()
def predict(model, images, idx, view='clean', batch=100):
    result = []
    for start in range(0, len(idx), batch):
        bi = idx[start:start+batch]
        x = kit.preprocess(images[bi])
        if view == 'bright': x = x*.8+.1
        elif view == 'crop': x = F.interpolate(x[...,13:-13,13:-13], size=224, mode='bilinear', align_corners=False)
        elif view == 'flip': x = x.flip(-1)
        elif view == 'blur': x = helpers.blur(x, 160)
        elif view == 'shift': x = F.pad(x,(8,8,8,8),mode='reflect')[...,5:229,11:235]
        elif view == 'noise':
            g = torch.Generator(device=x.device).manual_seed(700+start)
            x = x + .2*torch.randn(x.shape, device=x.device, generator=g)
        with torch.autocast('cuda', dtype=torch.float16):
            result.append(model(x).float().cpu())
    return torch.cat(result)


def append_result(name, strategy, result, checkpoint='', note=''):
    import fcntl
    c, a, r = result['clean'], result['bright'], result['control']
    row = [time.strftime('%d/%m %H:%M'), 'Codex', name, strategy,
           c['acc'], c['Lmean'], c['Lstd'], a['acc'], a['Lmean'], a['Lstd'],
           r['KL'], r['agree'], result.get('rel_dw',''), checkpoint, '', note]
    with (ROOT/'COORD/RESULTS.csv').open('a', newline='') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        csv.writer(handle).writerow(row)


@torch.no_grad()
def evaluate(model, original_state, imgs, proxy, proxy_logits, y, name, full=False):
    model.eval()
    idx = torch.arange(len(imgs)) if full else torch.arange(0,len(imgs),5)
    ctl = torch.arange(14000,16000)
    result = {'name':name, 'n_forget':len(idx), 'control_source':'aigen16k [14000:16000]; unlabeled, not true validation'}
    logits = {'indices':idx, 'y':y[idx]}
    for view in ['clean','bright','flip','crop','shift','noise','blur']:
        z = predict(model,imgs,idx,view)
        losses = F.cross_entropy(z,y[idx],reduction='none')
        result[view] = {'acc':(z.argmax(-1)==y[idx]).float().mean().item(),
                        'Lmean':losses.mean().item(), 'Lstd':losses.std().item()}
        logits[view] = z
    zc = predict(model,proxy,ctl)
    z0 = proxy_logits[ctl]
    hc = z0.softmax(-1).max(-1).values > .9
    result['control'] = {'KL':kl((z0/2).softmax(-1),zc).mean().item(),
                         'agree':(zc.argmax(-1)==z0.argmax(-1)).float().mean().item(),
                         'agree_conf':(zc.argmax(-1)[hc]==z0.argmax(-1)[hc]).float().mean().item(),
                         'n_conf':int(hc.sum())}
    logits['control'] = zc
    logits['control_original'] = z0
    delta = sum((p.float().cpu()-original_state[k]).square().sum().item() for k,p in model.state_dict().items())
    norm = sum(p.square().sum().item() for p in original_state.values())
    result['rel_dw'] = (delta/norm)**.5
    atomic_save(logits, OUT/'logs'/f'{name}_logits.pt')
    (OUT/'logs'/f'{name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result), flush=True)
    return result


def save_checkpoint(model, name):
    state = {k:v.detach().cpu().float().contiguous() for k,v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() and v.dtype == torch.float32 for v in state.values())
    path = OUT/'ckpt'/f'{name}.pt'
    atomic_save(state,path)
    # Validate the actual saved artifact, including exact architecture compatibility.
    verify = kit.build_model()
    verify.load_state_dict(load(path), strict=True)
    assert path.stat().st_size < 450_000_000
    print('CHECKPOINT_VALID',str(path),path.stat().st_size,flush=True)
    return str(path.relative_to(ROOT))
