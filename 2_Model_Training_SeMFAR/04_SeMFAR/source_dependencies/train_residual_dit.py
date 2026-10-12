import argparse,csv,json,math,pickle,random,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

WORKSPACE=Path(__file__).resolve().parents[1]
EDSR_ROOT=WORKSPACE/'EDSR'
sys.path.insert(0,str(WORKSPACE/'Difficulty_Ver1'))
from ema_teacher_core import EDSR,load_gray,ssim  # noqa

def discover(paired):
    out=[]
    for lr in sorted(paired.glob('*_LR.png')):
        name=lr.stem[:-3];hr=paired/f'{name}_HR.png'
        if hr.exists():out.append((name,lr,hr))
    return out

class Pairs(Dataset):
    def __init__(self,records,augment=False):self.records,self.augment=records,augment
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        name,lp,hp=self.records[i];lr,hr=load_gray(lp),load_gray(hp)
        if self.augment:
            k=random.randrange(4);flip=random.random()<.5;lr=torch.rot90(lr,k,(1,2));hr=torch.rot90(hr,k,(1,2))
            if flip:lr,hr=lr.flip(2),hr.flip(2)
        return lr,hr,name

def timestep_embedding(t,dim=256,max_period=10000):
    half=dim//2;freq=torch.exp(-math.log(max_period)*torch.arange(half,device=t.device)/half)
    args=t.float()[:,None]*freq[None];emb=torch.cat([torch.cos(args),torch.sin(args)],1)
    return F.pad(emb,(0,dim-emb.shape[1]))

def fixed_2d_position(size=32,dim=256):
    quarter=dim//4;y,x=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
    freq=torch.exp(-math.log(10000)*torch.arange(quarter)/quarter)
    ex=x.flatten().float()[:,None]*freq;ey=y.flatten().float()[:,None]*freq
    return torch.cat([torch.sin(ex),torch.cos(ex),torch.sin(ey),torch.cos(ey)],1)

class AdaLNBlock(nn.Module):
    def __init__(self,dim=256,heads=8,mlp_ratio=4):
        super().__init__();self.n1=nn.LayerNorm(dim,elementwise_affine=False);self.n2=nn.LayerNorm(dim,elementwise_affine=False)
        self.mod=nn.Sequential(nn.SiLU(),nn.Linear(dim,2*dim));self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*mlp_ratio),nn.GELU(),nn.Linear(dim*mlp_ratio,dim))
    def adaln(self,x,e):
        gamma,beta=self.mod(e).chunk(2,dim=-1);return gamma[:,None]*x+beta[:,None]
    def forward(self,x,e):
        q=self.adaln(self.n1(x),e);x=x+self.attn(q,q,q,need_weights=False)[0]
        return x+self.mlp(self.adaln(self.n2(x),e))

class ResidualDiT(nn.Module):
    def __init__(self,dim=256,depth=8,heads=8,patch=8):
        super().__init__();self.patch=patch
        self.r_conv=nn.Conv2d(1,64,3,padding=1);self.c_conv=nn.Conv2d(1,64,3,padding=1)
        self.r_patch=nn.Conv2d(64,dim,patch,stride=patch);self.c_patch=nn.Conv2d(64,dim,patch,stride=patch)
        self.register_buffer('position',fixed_2d_position(256//patch,dim),persistent=True)
        self.time=nn.Sequential(nn.Linear(dim,dim*4),nn.SiLU(),nn.Linear(dim*4,dim))
        self.blocks=nn.ModuleList([AdaLNBlock(dim,heads,4) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim);self.out=nn.Linear(dim,patch*patch)
    def forward(self,r_t,t,coarse):
        xr=self.r_patch(self.r_conv(r_t)).flatten(2).transpose(1,2);c=self.c_patch(self.c_conv(coarse)).flatten(2).transpose(1,2)
        x=xr+c+self.position[None].to(xr.dtype);e=self.time(timestep_embedding(t,x.shape[-1]).to(x.dtype))
        for block in self.blocks:x=block(x,e)
        p=self.patch;b,n,_=x.shape;x=self.out(self.norm(x)).view(b,32,32,p,p).permute(0,1,3,2,4).reshape(b,1,256,256)
        return x

class CosineDiffusion:
    def __init__(self,T=1000,device='cpu'):
        self.T=T;s=.008;t=torch.arange(T+1,device=device,dtype=torch.float64)
        ab=torch.cos(((t/T+s)/(1+s))*math.pi/2).square();self.ab=(ab/ab[0]).float()
    def q(self,x0,t,noise):
        a=self.ab[t].view(-1,1,1,1);return a.sqrt()*x0+(1-a).sqrt()*noise

@torch.no_grad()
def ddim(model,coarse,diff,steps=50):
    # The cosine endpoint alpha_bar[T] is zero; DDIM must start at T-1.
    b=coarse.shape[0];x=torch.randn_like(coarse);times=torch.linspace(diff.T-1,1,steps,device=coarse.device).round().long().unique_consecutive()
    for j,tval in enumerate(times):
        t=torch.full((b,),int(tval),device=coarse.device,dtype=torch.long);a=diff.ab[tval];eps=model(x,t,coarse)
        x0=((x-(1-a).sqrt()*eps)/a.sqrt().clamp_min(1e-6)).clamp(-1,1);prev=times[j+1] if j+1<len(times) else torch.tensor(0,device=coarse.device)
        ap=diff.ab[prev];x=ap.sqrt()*x0+(1-ap).sqrt()*eps
    return x

@torch.no_grad()
def val_loss(dit,edsr,loader,diff,device):
    dit.eval();total=count=0;g=torch.Generator(device=device).manual_seed(123456)
    for lr,hr,_ in loader:
        lr,hr=lr.to(device),hr.to(device);coarse=edsr(lr);target=hr-coarse;t=torch.randint(1,diff.T+1,(len(lr),),device=device,generator=g);noise=torch.randn(target.shape,device=device,generator=g);pred=dit(diff.q(target,t,noise),t,coarse)
        total+=F.mse_loss(pred,noise,reduction='sum').item();count+=noise.numel()
    return total/count

def gray(path):
    with Image.open(path) as im:return np.asarray(im.convert('L'),np.float32)/255
def ssim_map(a,b):
    a=torch.from_numpy(a)[None,None];b=torch.from_numpy(b)[None,None];ma,mb=F.avg_pool2d(a,11,1,5),F.avg_pool2d(b,11,1,5);va=F.avg_pool2d(a*a,11,1,5)-ma.square();vb=F.avg_pool2d(b*b,11,1,5)-mb.square();vab=F.avg_pool2d(a*b,11,1,5)-ma*mb
    return (((2*ma*mb+.01**2)*(2*vab+.03**2))/((ma.square()+mb.square()+.01**2)*(va+vb+.03**2)))[0,0].numpy()
def tumor_mask(paired,name):
    p=paired/f'{name}_HR.pkl';p=p if p.exists() else paired/f'{name}_LR.pkl'
    with p.open('rb') as f:s=pickle.load(f)['full_semantic_map'];m=s[:,:,s.shape[2]//2]==4
    return np.asarray(Image.fromarray(m).resize((256,256),Image.Resampling.NEAREST),bool)
def otsu(x):
    h,_=np.histogram(x,256,(0,1));p=h/h.sum();w=np.cumsum(p);m=np.cumsum(p*np.arange(256));v=(m[-1]*w-m)**2/(w*(1-w)+1e-12);return np.argmax(v)/255
def dice(a,b):
    d=a.sum()+b.sum();return 1. if d==0 else float(2*(a&b).sum()/d)

def evaluate_images(output,paired,names,best_epoch):
    rows=[]
    for name in names:
        pred=gray(output/'test_predictions'/f'{name}_pred.png');hr=gray(paired/f'{name}_HR.png');sm=ssim_map(pred,hr);roi=tumor_mask(paired,name);th=otsu(hr)
        rows.append({'file':name,'test_psnr_db':-10*np.log10(max(np.mean((pred-hr)**2),1e-12)),'test_ssim':sm.mean(),'tumor_roi_psnr_db':-10*np.log10(max(np.mean((pred[roi]-hr[roi])**2),1e-12)),'tumor_roi_ssim':sm[roi].mean(),'picture_mask_dice':dice(pred<th,hr<th)})
    with (output/'test_per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={'model':'Difficulty_Ver1_EDSR_plus_Small_Residual_DiT','best_epoch':best_epoch,'n':len(rows)}
    for k in rows[0]:
        if k=='file':continue
        v=np.asarray([r[k] for r in rows],float);summary[k]={'mean':float(v.mean()),'std':float(v.std(ddof=1))}
    (output/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paired',type=Path,default=WORKSPACE/'Paired');ap.add_argument('--split',type=Path,default=EDSR_ROOT/'Comparison_300epochs_CyclicUnpaired_4064'/'Difficulty_Ver1'/'split.json');ap.add_argument('--edsr-checkpoint',type=Path,default=EDSR_ROOT/'Comparison_300epochs_CyclicUnpaired_4064'/'Difficulty_Ver1'/'best.pt');ap.add_argument('--output',type=Path,default=WORKSPACE/'DiT'/'Residual_DiT_Difficulty_Ver1_300epochs')
    ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--ddim-steps',type=int,default=50);ap.add_argument('--eta',type=float,default=1.)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    split=json.loads(a.split.read_text());records={r[0]:r for r in discover(a.paired)};sets={k:[records[n] for n in split[k]] for k in ('train','validation','test')};(a.output/'split.json').write_text(json.dumps(split,indent=2))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(a.edsr_checkpoint,map_location=device,weights_only=False);scale=ck.get('args',{}).get('residual_scale',.1);edsr=EDSR(scale).to(device);edsr.load_state_dict(ck['student']);edsr.eval().requires_grad_(False)
    dit=ResidualDiT().to(device);opt=torch.optim.AdamW(dit.parameters(),lr=a.lr);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');diff=CosineDiffusion(1000,device)
    train=DataLoader(Pairs(sets['train'],True),a.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=='cuda');val=DataLoader(Pairs(sets['validation']),a.batch_size);test=DataLoader(Pairs(sets['test']),1)
    fields=['epoch','train_diffusion_mse','val_diffusion_mse'];best=float('inf')
    with (a.output/'losses.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for epoch in range(1,a.epochs+1):
            dit.train();total=count=0
            for lr,hr,_ in train:
                lr,hr=lr.to(device),hr.to(device);opt.zero_grad(set_to_none=True)
                with torch.no_grad():coarse=edsr(lr);target=hr-coarse
                t=torch.randint(1,diff.T+1,(len(lr),),device=device);noise=torch.randn_like(target);r_t=diff.q(target,t,noise)
                with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                    pred=dit(r_t,t,coarse);loss=F.mse_loss(pred,noise)
                scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=loss.item()*len(lr);count+=len(lr)
            vm=val_loss(dit,edsr,val,diff,device);row={'epoch':epoch,'train_diffusion_mse':total/count,'val_diffusion_mse':vm};w.writerow(row);f.flush();print(f'Epoch {epoch:03d}/{a.epochs} train_diff={row["train_diffusion_mse"]:.6f} val_diff={vm:.6f}',flush=True)
            state={'epoch':epoch,'dit':dit.state_dict(),'optimizer':opt.state_dict(),'args':vars(a),'edsr_checkpoint':str(a.edsr_checkpoint)};torch.save(state,a.output/'last.pt')
            if vm<best:best=vm;torch.save(state,a.output/'best.pt')
    state=torch.load(a.output/'best.pt',map_location=device,weights_only=False);dit.load_state_dict(state['dit']);dit.eval();pred_dir=a.output/'test_predictions';pred_dir.mkdir(exist_ok=True);panels=[]
    with torch.no_grad():
        for lr,hr,names in test:
            lr,hr=lr.to(device),hr.to(device);coarse=edsr(lr);res=ddim(dit,coarse,diff,a.ddim_steps);final=(coarse+a.eta*res).clamp(0,1);name=names[0]
            arrays=[(z[0,0].cpu().numpy()*255).round().astype(np.uint8) for z in (lr,coarse,final,hr)];Image.fromarray(arrays[2]).save(pred_dir/f'{name}_pred.png');canvas=Image.new('L',(1024,292),255);draw=ImageDraw.Draw(canvas)
            for i,(arr,label) in enumerate(zip(arrays,('LR input','Frozen EDSR','EDSR + Residual DiT','HR ground truth'))):canvas.paste(Image.fromarray(arr),(i*256,36));draw.text((i*256+8,10),label,fill=0)
            draw.text((8,274),name,fill=0);canvas.save(pred_dir/f'{name}_comparison.png');panels.append(canvas)
    sheet=Image.new('L',(1024,292*len(panels)),255)
    for i,p in enumerate(panels):sheet.paste(p,(0,i*292))
    sheet.save(a.output/'test_comparison_all.png');summary=evaluate_images(a.output,a.paired,split['test'],state['epoch']);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
