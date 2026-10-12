"""Ver7: deterministic one-pass residual Transformer; no diffusion, noise, or timestep."""
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from train_residual_dit import WORKSPACE,EDSR_ROOT,EDSR,Pairs,discover,evaluate_images,fixed_2d_position
from train_residual_dit_ver4 import charbonnier
from train_residual_dit_ver5 import pooled_corr

class TransformerBlock(nn.Module):
    def __init__(self,dim=256,heads=8,mlp_ratio=4):
        super().__init__();self.n1=nn.LayerNorm(dim);self.n2=nn.LayerNorm(dim);self.attn=nn.MultiheadAttention(dim,heads,batch_first=True);self.mlp=nn.Sequential(nn.Linear(dim,dim*mlp_ratio),nn.GELU(),nn.Linear(dim*mlp_ratio,dim))
    def forward(self,x):
        q=self.n1(x);x=x+self.attn(q,q,q,need_weights=False)[0];return x+self.mlp(self.n2(x))

class DeterministicResidualTransformer(nn.Module):
    def __init__(self,dim=256,depth=8,heads=8,patch=8):
        super().__init__();self.patch=patch;self.lr_conv=nn.Conv2d(1,64,3,padding=1);self.coarse_conv=nn.Conv2d(1,64,3,padding=1);self.lr_patch=nn.Conv2d(64,dim,patch,stride=patch);self.coarse_patch=nn.Conv2d(64,dim,patch,stride=patch);self.q_norm=nn.LayerNorm(dim);self.c_norm=nn.LayerNorm(dim);self.cross=nn.MultiheadAttention(dim,heads,batch_first=True);self.register_buffer('position',fixed_2d_position(256//patch,dim),persistent=True);self.blocks=nn.ModuleList([TransformerBlock(dim,heads,4) for _ in range(depth)]);self.norm=nn.LayerNorm(dim);self.out=nn.Linear(dim,patch*patch)
    def forward(self,lr,coarse):
        q=self.lr_patch(self.lr_conv(lr)).flatten(2).transpose(1,2);c=self.coarse_patch(self.coarse_conv(coarse)).flatten(2).transpose(1,2);x=self.cross(self.q_norm(q),self.c_norm(c),self.c_norm(c),need_weights=False)[0]+self.position[None].to(q.dtype)
        for block in self.blocks:x=block(x)
        p=self.patch;b=x.shape[0];return self.out(self.norm(x)).view(b,32,32,p,p).permute(0,1,3,2,4).reshape(b,1,256,256)

@torch.no_grad()
def validate(model,edsr,loader,device,eps):
    model.eval();chars=[];ps=[];ys=[]
    for lr,hr,_ in loader:
        lr,hr=lr.to(device),hr.to(device);coarse=edsr(lr);target=hr-coarse;pred=model(lr,coarse);chars.append(charbonnier(pred,target,eps).item()*len(lr));ps.append(pred.cpu().flatten());ys.append(target.cpu().flatten())
    p,y=torch.cat(ps),torch.cat(ys);return {'val_charbonnier':sum(chars)/len(loader.dataset),'val_residual_mse':float(F.mse_loss(p,y)),'val_residual_mae':float(F.l1_loss(p,y)),'val_residual_corr':pooled_corr(p,y),'val_pred_std':float(p.std(unbiased=False)),'val_gt_std':float(y.std(unbiased=False)),'val_std_ratio':float(p.std(unbiased=False)/y.std(unbiased=False).clamp_min(1e-12))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paired',type=Path,default=WORKSPACE/'Paired');ap.add_argument('--split',type=Path,default=EDSR_ROOT/'Comparison_300epochs_CyclicUnpaired_4064'/'Difficulty_Ver1'/'split.json');ap.add_argument('--edsr-checkpoint',type=Path,default=EDSR_ROOT/'Comparison_300epochs_CyclicUnpaired_4064'/'Difficulty_Ver1'/'best.pt');ap.add_argument('--output',type=Path,default=HERE/'Residual_Transformer_Ver7_Deterministic_300epochs');ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--eta',type=float,default=1.);ap.add_argument('--charbonnier-eps',type=float,default=1e-3)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);split=json.loads(a.split.read_text());records={r[0]:r for r in discover(a.paired)};sets={k:[records[n] for n in split[k]] for k in ('train','validation','test')};(a.output/'split.json').write_text(json.dumps(split,indent=2));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(a.edsr_checkpoint,map_location=device,weights_only=False);edsr=EDSR(ck.get('args',{}).get('residual_scale',.1)).to(device);edsr.load_state_dict(ck['student']);edsr.eval().requires_grad_(False);model=DeterministicResidualTransformer().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');train=DataLoader(Pairs(sets['train'],True),a.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=='cuda');val=DataLoader(Pairs(sets['validation']),a.batch_size);test=DataLoader(Pairs(sets['test']),1);fields=['epoch','train_charbonnier','val_charbonnier','val_residual_mse','val_residual_mae','val_residual_corr','val_pred_std','val_gt_std','val_std_ratio'];best=float('inf')
    with (a.output/'losses_and_validation_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for epoch in range(1,a.epochs+1):
            model.train();total=count=0
            for lr,hr,_ in train:
                lr,hr=lr.to(device),hr.to(device);opt.zero_grad(set_to_none=True)
                with torch.no_grad():coarse=edsr(lr);target=hr-coarse
                with torch.amp.autocast('cuda',enabled=device.type=='cuda'):loss=charbonnier(model(lr,coarse),target,a.charbonnier_eps)
                scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=loss.item()*len(lr);count+=len(lr)
            vd=validate(model,edsr,val,device,a.charbonnier_eps);row={'epoch':epoch,'train_charbonnier':total/count,**vd};w.writerow(row);f.flush();print(f'Epoch {epoch:03d}/{a.epochs} train={total/count:.6f} val={vd["val_charbonnier"]:.6f} MSE={vd["val_residual_mse"]:.6f} corr={vd["val_residual_corr"]:.4f} std={vd["val_pred_std"]:.4f}/{vd["val_gt_std"]:.4f}',flush=True);state={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'args':vars(a),'edsr_checkpoint':str(a.edsr_checkpoint),'prediction_type':'deterministic_residual','val_residual_mse':vd['val_residual_mse']};torch.save(state,a.output/'last.pt')
            if vd['val_residual_mse']<best:best=vd['val_residual_mse'];torch.save(state,a.output/'best.pt')
    state=torch.load(a.output/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);model.eval();pred_dir=a.output/'test_predictions';pred_dir.mkdir(exist_ok=True);diag=a.output/'residual_diagnostics';diag.mkdir(exist_ok=True);panels=[];res_panels=[];rows=[];all_gt=[];all_pred=[]
    with torch.no_grad():
        for lr,hr,names in test:
            lr,hr=lr.to(device),hr.to(device);coarse=edsr(lr);res=model(lr,coarse);final=(coarse+a.eta*res).clamp(0,1);name=names[0];g=(hr-coarse)[0,0].cpu().numpy();p=res[0,0].cpu().numpy();all_gt.append(g.ravel());all_pred.append(p.ravel());e=p-g;rows.append({'file':name,'residual_mse':float(np.mean(e**2)),'residual_mae':float(np.mean(np.abs(e))),'residual_corr':pooled_corr(torch.from_numpy(p),torch.from_numpy(g)),'std_R_GT':float(g.std()),'std_R_pred':float(p.std()),'std_ratio':float(p.std()/max(g.std(),1e-12))});arrays=[(z[0,0].cpu().numpy()*255).clip(0,255).round().astype(np.uint8) for z in (lr,coarse,final,hr)];Image.fromarray(arrays[2]).save(pred_dir/f'{name}_pred.png');canvas=Image.new('L',(1024,292),255);draw=ImageDraw.Draw(canvas)
            for i,(arr,label) in enumerate(zip(arrays,('LR input','Frozen Difficulty EDSR','EDSR + Ver7 residual','HR ground truth'))):canvas.paste(Image.fromarray(arr),(i*256,36));draw.text((i*256+8,10),label,fill=0)
            draw.text((8,274),name,fill=0);canvas.save(pred_dir/f'{name}_comparison.png');panels.append(canvas);signed=lambda x:np.clip((x/.8+.5)*255,0,255).round().astype(np.uint8);rc=Image.new('L',(768,292),255);rd=ImageDraw.Draw(rc)
            for i,(arr,label) in enumerate(zip((signed(g),signed(p),np.clip(np.abs(e)*255,0,255).astype(np.uint8)),('R_GT','R_hat','absolute error'))):rc.paste(Image.fromarray(arr),(i*256,36));rd.text((i*256+8,10),label,fill=0)
            rd.text((8,274),f'{name} MSE={rows[-1]["residual_mse"]:.5f} corr={rows[-1]["residual_corr"]:.4f}',fill=0);rc.save(diag/f'{name}_residual_comparison.png');res_panels.append(rc)
    sheet=Image.new('L',(1024,292*len(panels)),255);rsheet=Image.new('L',(768,292*len(res_panels)),255)
    for i,p in enumerate(panels):sheet.paste(p,(0,i*292))
    for i,p in enumerate(res_panels):rsheet.paste(p,(0,i*292))
    sheet.save(a.output/'test_comparison_all.png');rsheet.save(diag/'test_residual_comparison_all.png');summary=evaluate_images(a.output,a.paired,split['test'],state['epoch']);summary.update({'model':'Residual_Transformer_Ver7_Deterministic','checkpoint_metric':'minimum validation residual MSE'});(a.output/'summary.json').write_text(json.dumps(summary,indent=2));gt=np.concatenate(all_gt);pred=np.concatenate(all_pred);mse=float(np.mean((pred-gt)**2));zero=float(np.mean(gt**2));rsummary={'n_images':len(rows),'pooled':{'mse':mse,'mae':float(np.mean(np.abs(pred-gt))),'correlation':pooled_corr(torch.from_numpy(pred),torch.from_numpy(gt)),'std_R_GT':float(gt.std()),'std_R_DiT':float(pred.std()),'std_ratio_pred_to_gt':float(pred.std()/gt.std()),'zero_correction_mse':zero,'dit_error_to_zero_error_ratio':mse/zero}};(diag/'residual_metrics_summary.json').write_text(json.dumps(rsummary,indent=2))
    with (diag/'per_image_residual_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps({'test':summary,'residual':rsummary},indent=2));(a.output/'train_residual_transformer_ver7_frozen.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
if __name__=='__main__':main()
