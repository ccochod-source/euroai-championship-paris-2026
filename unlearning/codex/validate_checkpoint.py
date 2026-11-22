"""CPU validation of the exact on-disk state_dict; never reads credentials."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse
import hashlib
import json
import sys
import torch


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('checkpoint'); a=ap.parse_args()
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    import task_template as kit
    path=(root/a.checkpoint).resolve()
    size=path.stat().st_size
    assert 0<size<450_000_000,size
    sd=torch.load(path,map_location='cpu',weights_only=True)
    assert isinstance(sd,dict) and len(sd)>0
    assert all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 for v in sd.values())
    assert all(torch.isfinite(v).all().item() for v in sd.values())
    model=kit.build_model()
    result=model.load_state_dict(sd,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    report={'path':str(path),'bytes':size,'sha256':digest.hexdigest(),
            'all_float32':True,'all_finite':True,'strict_load':True,
            'architecture':'task_template.build_model(): vit_base_patch16_224 num_classes=100 init_values=1e-4',
            'note':'Format validation only; hidden validation accuracy unknown.'}
    (root/'codex'/f'validation_{path.stem}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
