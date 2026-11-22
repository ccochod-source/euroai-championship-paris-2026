import torch, timm, numpy as np, json, time
import task_template as kit
torch.set_num_threads(__import__('os').cpu_count())
x=kit.load_forget_images(); x=torch.as_tensor(np.asarray(x))
m=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=True,num_classes=0,img_size=224).eval()
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
out=[];t=time.time()
with torch.inference_mode():
    for i in range(0,len(x),64):
        b=(x[i:i+64].permute(0,3,1,2).float()/255-mean)/std
        out.append(m(b))
        if i%640==0: print(i,round(time.time()-t),flush=True)
torch.save(torch.cat(out),'loop/dino-forget.pt'); print('done',flush=True)

