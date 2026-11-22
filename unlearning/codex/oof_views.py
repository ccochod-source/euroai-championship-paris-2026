"""Shared, deterministic views. Models always receive their own normalization."""
import io
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

TRAIN_VIEWS = ('clean', 'flip', 'crop', 'bright', 'blur', 'color', 'jpeg')
EVAL_VIEWS = TRAIN_VIEWS + ('shift', 'rotate', 'noise')


def view(x, name):
    # x is RGB float in [0,1]. No random state, allowing exact cached targets.
    if name == 'clean': return x
    if name == 'flip': return x.flip(-1)
    if name == 'crop':
        return F.interpolate(x[..., 20:208, 11:199], size=224, mode='bilinear', align_corners=False)
    if name == 'bright': return (x * .8 + .15).clamp(0, 1)
    if name == 'blur':
        return F.interpolate(F.interpolate(x, size=112, mode='bilinear', align_corners=False), size=224, mode='bilinear', align_corners=False)
    if name == 'color':
        gray=(x*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
        return (.55*x+.45*gray).mul(1.12).sub(.06).clamp(0,1)
    if name == 'shift': return F.pad(x,(8,8,8,8),mode='reflect')[...,5:229,11:235]
    if name == 'rotate':
        angle = 5 * torch.pi / 180
        c,s=np.cos(angle),np.sin(angle)
        theta=x.new_tensor([[c,-s,0],[s,c,0]]).expand(len(x),-1,-1)
        return F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),padding_mode='reflection',align_corners=False)
    if name == 'noise':
        # Shared deterministic noise pattern makes batching irrelevant.
        gen=torch.Generator(device=x.device).manual_seed(90173)
        return (x+.075*torch.randn((1,3,224,224),generator=gen,device=x.device)).clamp(0,1)
    if name == 'jpeg':
        out=[]
        for im in (x.detach().cpu().permute(0,2,3,1)*255).round().byte().numpy():
            b=io.BytesIO(); Image.fromarray(im).save(b,format='JPEG',quality=50)
            b.seek(0); out.append(np.asarray(Image.open(b).convert('RGB')).copy())
        return torch.from_numpy(np.stack(out)).permute(0,3,1,2).to(x.device).float()/255
    raise ValueError(name)
