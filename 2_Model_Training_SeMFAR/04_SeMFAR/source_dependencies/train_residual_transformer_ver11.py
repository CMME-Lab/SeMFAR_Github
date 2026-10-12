"""Ver11 frequency-aware full-residual Transformer (A/B/C via lambda_full)."""
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;DIT=ROOT/'DiT';sys.path.insert(0,str(DIT))
from train_residual_dit import EDSR_ROOT,EDSR,Pairs,discover,evaluate_images
from train_residual_dit_ver4 import charbonnier
from train_residual_dit_ver5 import pooled_corr
from train_residual_transformer_ver7 import DeterministicResidualTransformer
def gaussian(x,s):
 r=int(np.ceil(3*s));t=torch.arange(-r,r+1,device=x.device,dtype=x.dtype);k=torch.exp(-.5*(t/s)**2);k/=k.sum();return F.conv2d(F.pad(x,(r,r,r,r),mode='reflect'),(k[:,None]*k[None,:])[None,None])
@torch.no_grad()
def validate(m,e,loader,d,eps,sigma,lf):
 m.eval();s=np.zeros(3);n=0;pl=[];yl=[];pf=[];yf=[]
 for lr,hr,_ in loader:
  lr,hr=lr.to(d),hr.to(d);c=e(lr);y=hr-c;p=m(lr,c);ylow=gaussian(y,sigma);plow=gaussian(p,sigma);ll=charbonnier(plow,ylow,eps);lfull=charbonnier(p,y,eps);total=ll+lf*lfull;s+=np.array([total.item(),ll.item(),lfull.item()])*len(lr);n+=len(lr);pl.append(plow.cpu().flatten());yl.append(ylow.cpu().flatten());pf.append(p.cpu().flatten());yf.append(y.cpu().flatten())
 pl,yl,pf,yf=map(torch.cat,(pl,yl,pf,yf));return {'val_total':s[0]/n,'val_low_charbonnier':s[1]/n,'val_full_charbonnier':s[2]/n,'val_low_mse':float(F.mse_loss(pl,yl)),'val_low_corr':pooled_corr(pl,yl),'val_low_pred_std':float(pl.std(unbiased=False)),'val_low_gt_std':float(yl.std(unbiased=False)),'val_low_std_ratio':float(pl.std(unbiased=False)/yl.std(unbiased=False).clamp_min(1e-12)),'val_full_mse':float(F.mse_loss(pf,yf)),'val_full_corr':pooled_corr(pf,yf)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--paired',type=Path,default=ROOT/'Paired');ap.add_argument('--split',type=Path,default=EDSR_ROOT/'Comparison_300epochs_CyclicUnpaired_4064'/'Difficulty_Ver1'/'split.json');ap.add_argument('--edsr-checkpoint',type=Path,default=EDSR_ROOT/'Comparison_300epochs_CyclicUnpaired_4064'/'Difficulty_Ver1'/'best.pt');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--variant',required=True);ap.add_argument('--lambda-full',type=float,required=True);ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--sigma',type=float,default=1.5);ap.add_argument('--eta-low',type=float,default=.75);ap.add_argument('--eta-high',type=float,default=0.);ap.add_argument('--charbonnier-eps',type=float,default=1e-3)
 a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);split=json.loads(a.split.read_text());rec={r[0]:r for r in discover(a.paired)};sets={k:[rec[n] for n in split[k]] for k in ('train','validation','test')};(a.output/'split.json').write_text(json.dumps(split,indent=2));d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(a.edsr_checkpoint,map_location=d,weights_only=False);e=EDSR(ck.get('args',{}).get('residual_scale',.1)).to(d);e.load_state_dict(ck['student']);e.eval().requires_grad_(False);m=DeterministicResidualTransformer().to(d);opt=torch.optim.AdamW(m.parameters(),lr=a.lr);scaler=torch.amp.GradScaler('cuda',enabled=d.type=='cuda');train=DataLoader(Pairs(sets['train'],True),a.batch_size,shuffle=True,num_workers=0,pin_memory=d.type=='cuda');val=DataLoader(Pairs(sets['validation']),a.batch_size);test=DataLoader(Pairs(sets['test']),1);vf=['val_total','val_low_charbonnier','val_full_charbonnier','val_low_mse','val_low_corr','val_low_pred_std','val_low_gt_std','val_low_std_ratio','val_full_mse','val_full_corr'];fields=['epoch','train_total','train_low_charbonnier','train_full_charbonnier',*vf];best=float('inf')
 with (a.output/'losses_and_validation_diagnostics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for ep in range(1,a.epochs+1):
   m.train();s=np.zeros(3);n=0
   for lr,hr,_ in train:
    lr,hr=lr.to(d),hr.to(d);opt.zero_grad(set_to_none=True)
    with torch.no_grad():c=e(lr);y=hr-c;yl=gaussian(y,a.sigma)
    with torch.amp.autocast('cuda',enabled=d.type=='cuda'):p=m(lr,c);low=charbonnier(gaussian(p,a.sigma),yl,a.charbonnier_eps);full=charbonnier(p,y,a.charbonnier_eps);loss=low+a.lambda_full*full
    scaler.scale(loss).backward();scaler.step(opt);scaler.update();s+=np.array([loss.item(),low.item(),full.item()])*len(lr);n+=len(lr)
   tr=s/n;vd=validate(m,e,val,d,a.charbonnier_eps,a.sigma,a.lambda_full);w.writerow(dict(zip(fields,[ep,*tr,*vd.values()])));f.flush();print(f'{a.variant} Epoch {ep:03d}/{a.epochs} train={tr[0]:.6f} low={tr[1]:.6f} full={tr[2]:.6f} | val={vd["val_total"]:.6f} low-corr={vd["val_low_corr"]:.4f} ratio={vd["val_low_std_ratio"]:.3f} full-corr={vd["val_full_corr"]:.4f}',flush=True);state={'epoch':ep,'model':m.state_dict(),'optimizer':opt.state_dict(),'args':vars(a),'edsr_checkpoint':str(a.edsr_checkpoint),'val_total':vd['val_total']};torch.save(state,a.output/'last.pt')
   if vd['val_total']<best:best=vd['val_total'];torch.save(state,a.output/'best.pt')
 state=torch.load(a.output/'best.pt',map_location=d,weights_only=False);m.load_state_dict(state['model']);m.eval();pd=a.output/'test_predictions';pd.mkdir(exist_ok=True);diag=a.output/'residual_diagnostics';diag.mkdir(exist_ok=True);panels=[];rpanels=[];rows=[];allpl=[];allyl=[];allp=[];ally=[]
 with torch.no_grad():
  for lr,hr,names in test:
   lr,hr=lr.to(d),hr.to(d);c=e(lr);y=hr-c;p=m(lr,c);pl=gaussian(p,a.sigma);ph=p-pl;applied=a.eta_low*pl+a.eta_high*ph;final=(c+applied).clamp(0,1);yl=gaussian(y,a.sigma);name=names[0];yn=y[0,0].cpu().numpy();yln=yl[0,0].cpu().numpy();pn=p[0,0].cpu().numpy();pln=pl[0,0].cpu().numpy();allpl.append(pln.ravel());allyl.append(yln.ravel());allp.append(pn.ravel());ally.append(yn.ravel());rows.append({'file':name,'low_mse':float(np.mean((pln-yln)**2)),'low_mae':float(np.mean(np.abs(pln-yln))),'low_corr':pooled_corr(torch.from_numpy(pln),torch.from_numpy(yln)),'low_std_pred':float(pln.std()),'low_std_gt':float(yln.std()),'low_std_ratio':float(pln.std()/yln.std()),'full_mse':float(np.mean((pn-yn)**2)),'full_corr':pooled_corr(torch.from_numpy(pn),torch.from_numpy(yn))});arr=[(z[0,0].cpu().numpy()*255).clip(0,255).round().astype(np.uint8) for z in (lr,c,final,hr)];Image.fromarray(arr[2]).save(pd/f'{name}_pred.png');cv=Image.new('L',(1024,292),255);dr=ImageDraw.Draw(cv)
   for i,(x,label) in enumerate(zip(arr,('LR input','Frozen Difficulty EDSR',f'EDSR + Ver11-{a.variant}','HR ground truth'))):cv.paste(Image.fromarray(x),(i*256,36));dr.text((i*256+8,10),label,fill=0)
   dr.text((8,274),name,fill=0);cv.save(pd/f'{name}_comparison.png');panels.append(cv);signed=lambda x:np.clip((x/.8+.5)*255,0,255).round().astype(np.uint8);rc=Image.new('L',(1024,292),255);rd=ImageDraw.Draw(rc)
   for i,(x,label) in enumerate(zip((signed(yn),signed(yln),signed(pn),signed(pln)),('Full R_GT','Low R_GT','Raw R_hat','Low R_hat'))):rc.paste(Image.fromarray(x),(i*256,36));rd.text((i*256+8,10),label,fill=0)
   rd.text((8,274),f'{name} low corr={rows[-1]["low_corr"]:.4f} ratio={rows[-1]["low_std_ratio"]:.3f}',fill=0);rc.save(diag/f'{name}_residual_comparison.png');rpanels.append(rc)
 sh=Image.new('L',(1024,292*len(panels)),255);rh=Image.new('L',(1024,292*len(rpanels)),255)
 for i,x in enumerate(panels):sh.paste(x,(0,i*292))
 for i,x in enumerate(rpanels):rh.paste(x,(0,i*292))
 sh.save(a.output/'test_comparison_all.png');rh.save(diag/'test_residual_comparison_all.png');summary=evaluate_images(a.output,a.paired,split['test'],state['epoch']);summary.update({'model':f'Ver11-{a.variant}','sigma':a.sigma,'lambda_full':a.lambda_full,'eta_low':a.eta_low,'eta_high':a.eta_high,'checkpoint_metric':'minimum validation total loss'});(a.output/'summary.json').write_text(json.dumps(summary,indent=2));pl,yl,p,y=map(np.concatenate,(allpl,allyl,allp,ally));lm=float(np.mean((pl-yl)**2));zm=float(np.mean(yl**2));rs={'n_images':len(rows),'pooled':{'low_mse':lm,'low_mae':float(np.mean(np.abs(pl-yl))),'low_correlation':pooled_corr(torch.from_numpy(pl),torch.from_numpy(yl)),'low_std_pred':float(pl.std()),'low_std_gt':float(yl.std()),'low_std_ratio':float(pl.std()/yl.std()),'low_zero_mse':zm,'low_error_to_zero_ratio':lm/zm,'full_mse':float(np.mean((p-y)**2)),'full_correlation':pooled_corr(torch.from_numpy(p),torch.from_numpy(y)),'full_std_pred':float(p.std()),'full_std_gt':float(y.std())}};(diag/'residual_metrics_summary.json').write_text(json.dumps(rs,indent=2))
 with (diag/'per_image_residual_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (a.output/'architecture_and_training_config.json').write_text(json.dumps({'architecture':'Ver7 full residual Transformer','sigma':a.sigma,'loss':f'Char(G(Rhat),G(RGT))+{a.lambda_full}*Char(Rhat,RGT)','inference':f'EDSR+{a.eta_low}*G(Rhat)+{a.eta_high}*(Rhat-G(Rhat))','epochs':a.epochs},indent=2));(a.output/'train_residual_transformer_ver11_frozen.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8');print(json.dumps({'test':summary,'residual':rs},indent=2))
if __name__=='__main__':main()
