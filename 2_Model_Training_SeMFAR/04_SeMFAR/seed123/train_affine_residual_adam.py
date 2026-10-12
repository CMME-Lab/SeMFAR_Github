import argparse, csv, json, random, shutil, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT=Path(r"D:\Woojeong_SR")
VER7=ROOT/"Comparison_300epochs_Data_ver10_50paired_1018unpaired_2"
sys.path[:0]=[str(VER7/"DiT"),str(ROOT/"Deterministic_Residual_Transformer"),str(ROOT/"Foundation_Model_1")]
from train_residual_dit import EDSR, Pairs, discover, tumor_mask
from train_residual_transformer_ver7 import DeterministicResidualTransformer
from train_residual_transformer_ver11 import gaussian
from train_local_scale_ver16 import frozen, corr, metric, summarize
from train_residual_transformer_ver18_echocare import make_echocare

V11=VER7/"Residual_Transformer_Ver11_A_eta1"
V16=VER7/"Local_Scale_Ver16_A_Ridge"
UPSTREAM=VER7/"Paired200_Beta2_ROI_Upstream"/"best.pt"
PAIRED=ROOT/"data10"/"Paired_ver10"
EPS=1e-12


def ridge_target_patch(gt,pred,ridge,lo,hi,patch):
    numerator=F.avg_pool2d(gt*pred,patch,patch)
    energy=F.avg_pool2d(pred.square(),patch,patch)
    return (numerator/(energy+ridge)).clamp(lo,hi),energy


def gradient_char(pred,target,eps=1e-3):
    px,tx=pred[:,:,:,1:]-pred[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1]
    py,ty=pred[:,:,1:,:]-pred[:,:,:-1,:],target[:,:,1:,:]-target[:,:,:-1,:]
    return .5*(torch.sqrt((px-tx).square()+eps*eps).mean()+torch.sqrt((py-ty).square()+eps*eps).mean())


class StructureNet(nn.Module):
    def __init__(self,variant):
        super().__init__(); self.variant=variant
        if variant=="multiscale_f1_f2":
            self.p1=nn.Sequential(nn.Conv2d(128,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1),nn.GELU())
            self.p2=nn.Sequential(nn.Conv2d(256,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1),nn.GELU())
            cin,out=65,1
        else:
            self.p2=nn.Sequential(nn.Conv2d(256,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GELU())
            cin=66 if variant=="boundary_dual" else 65
            out=2 if variant in ("boundary_dual","affine") else 1
        self.fuse=nn.Sequential(nn.Conv2d(cin,64,3,padding=1),nn.GELU(),
                                nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,out,1))

    def forward(self,feats,pred):
        if self.variant=="highres_4x4":
            e=F.adaptive_avg_pool2d(self.p2(feats[1]),(64,64)); r=F.avg_pool2d(pred,4,4)
            return self.fuse(torch.cat((e,r),1))
        if self.variant=="multiscale_f1_f2":
            e1=F.adaptive_avg_pool2d(self.p1(feats[0]),(32,32))
            e2=F.adaptive_avg_pool2d(self.p2(feats[1]),(32,32)); r=F.avg_pool2d(pred,8,8)
            return self.fuse(torch.cat((e1,e2,r),1))
        e=F.adaptive_avg_pool2d(self.p2(feats[1]),(32,32))
        if self.variant=="boundary_dual":
            low=gaussian(pred,2.0); edge=pred-low
            return self.fuse(torch.cat((e,F.avg_pool2d(low,8,8),F.avg_pool2d(edge,8,8)),1))
        return self.fuse(torch.cat((e,F.avg_pool2d(pred,8,8)),1))


@torch.no_grad()
def echo_features(echo,lr):
    return echo(lr.repeat(1,3,1,1))


def predict_and_targets(net,variant,feats,pred,gt,ridge,lo,hi):
    raw=net(feats,pred)
    if variant=="boundary_dual":
        lowp=gaussian(pred,2.0); edgep=pred-lowp; lowgt=gaussian(gt,2.0); edgegt=gt-lowgt
        tl,_=ridge_target_patch(lowgt,lowp,ridge,lo,hi,8); te,_=ridge_target_patch(edgegt,edgep,ridge,lo,hi,8)
        scales=lo+(hi-lo)*torch.sigmoid(raw)
        final_res=F.interpolate(scales[:,0:1],size=pred.shape[-2:],mode="nearest")*lowp
        final_res+=F.interpolate(scales[:,1:2],size=pred.shape[-2:],mode="nearest")*edgep
        scale_loss=.5*(F.smooth_l1_loss(scales[:,0:1],tl)+F.smooth_l1_loss(scales[:,1:2],te))
        target=torch.cat((tl,te),1); return final_res,scales,target,scale_loss
    patch=4 if variant=="highres_4x4" else 8
    target,_=ridge_target_patch(gt,pred,ridge,lo,hi,patch)
    if variant=="affine":
        scale=lo+(hi-lo)*torch.sigmoid(raw[:,0:1]); bias=.1*torch.tanh(raw[:,1:2])
        dense_a=F.interpolate(scale,size=pred.shape[-2:],mode="nearest")
        dense_b=F.interpolate(bias,size=pred.shape[-2:],mode="nearest")
        return dense_a*pred+dense_b,torch.cat((scale,bias),1),target,F.smooth_l1_loss(scale,target)
    scale=lo+(hi-lo)*torch.sigmoid(raw)
    return F.interpolate(scale,size=pred.shape[-2:],mode="nearest")*pred,scale,target,F.smooth_l1_loss(scale,target)


@torch.no_grad()
def validate(net,variant,edsr,residual,echo,loader,device,ridge,lo,hi):
    net.eval(); sums=np.zeros(6); count=0; pp=[]; yy=[]
    for lr,hr,_ in loader:
        lr,hr=lr.to(device),hr.to(device); base,pred,gt=frozen(lr,hr,edsr,residual)
        final_res,params,target,scale_loss=predict_and_targets(net,variant,echo_features(echo,lr),pred,gt,ridge,lo,hi)
        final=(base+final_res).clamp(0,1); grad=.2*gradient_char(final,hr); total=scale_loss+grad; n=len(lr)
        tracked=params[:,:target.shape[1]]
        sums+=np.array([scale_loss.item(),grad.item(),total.item(),F.mse_loss(tracked,target).item(),F.l1_loss(tracked,target).item(),n])*n
        pp.append(tracked.cpu()); yy.append(target.cpu()); count+=n
    pp,yy=torch.cat(pp),torch.cat(yy)
    return {"val_scale_loss":sums[0]/sums[5],"val_gradient_term":sums[1]/sums[5],"val_total":sums[2]/sums[5],
            "val_scale_mse":sums[3]/sums[5],"val_scale_mae":sums[4]/sums[5],"val_scale_corr":corr(pp,yy),
            "val_scale_mean":float(pp.mean()),"val_target_mean":float(yy.mean())}


def plot(csv_path,out):
    d=np.genfromtxt(csv_path,delimiter=",",names=True); fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot(d["epoch"],d["train_total"],label="train total"); axes[0].plot(d["epoch"],d["val_total"],label="val total")
    axes[1].plot(d["epoch"],d["train_scale_loss"],label="train scale"); axes[1].plot(d["epoch"],d["val_scale_loss"],label="val scale")
    axes[2].plot(d["epoch"],d["train_gradient_term"],label="train 0.2 grad"); axes[2].plot(d["epoch"],d["val_gradient_term"],label="val 0.2 grad")
    for ax,title in zip(axes,("Total loss","Scale loss","Gradient term")): ax.set_title(title); ax.grid(alpha=.25); ax.legend(); ax.set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(out/"loss_curves.png",dpi=180); fig.savefig(out/"loss_curves.pdf"); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--variant",required=True,choices=("boundary_dual","highres_4x4","multiscale_f1_f2","affine"))
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--epochs",type=int,default=300)
    ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--effective-batch-size",type=int,default=16)
    ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True); accum=a.effective_batch_size//a.batch_size
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state11=torch.load(V11/"best.pt",map_location=device,weights_only=False); split=json.loads((V11/"split.json").read_text())
    rec={x[0]:x for x in discover(PAIRED)}; sets={k:[rec[n] for n in split[k]] for k in ("train","validation","test")}
    (a.output/"split.json").write_text(json.dumps(split,indent=2))
    ck=torch.load(UPSTREAM,map_location=device,weights_only=False); edsr=EDSR(ck.get("args",{}).get("residual_scale",.1)).to(device)
    edsr.load_state_dict(ck["student"]); edsr.eval().requires_grad_(False)
    residual=DeterministicResidualTransformer().to(device); residual.load_state_dict(state11["model"]); residual.eval().requires_grad_(False)
    echo=make_echocare(device); cfg=json.loads((V16/"ridge_and_target_config.json").read_text()); ridge=float(cfg["ridge_lambda"]); lo,hi=map(float,cfg["scale_bounds"])
    tr=DataLoader(Pairs(sets["train"],True),a.batch_size,shuffle=True,num_workers=0); va=DataLoader(Pairs(sets["validation"]),a.batch_size,num_workers=0); te=DataLoader(Pairs(sets["test"]),1,num_workers=0)
    net=StructureNet(a.variant).to(device); opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    fields=["epoch","train_scale_loss","train_gradient_term","train_total","val_scale_loss","val_gradient_term","val_total","val_scale_mse","val_scale_mae","val_scale_corr","val_scale_mean","val_target_mean"]
    best=float("inf"); best_epoch=0; early_best=float("inf"); early_epoch=0
    with (a.output/"losses_and_scale_diagnostics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for epoch in range(1,a.epochs+1):
            net.train(); opt.zero_grad(set_to_none=True); sums=np.zeros(3); count=0
            for step,(lr,hr,_) in enumerate(tr,1):
                lr,hr=lr.to(device),hr.to(device)
                with torch.no_grad(): base,pred,gt=frozen(lr,hr,edsr,residual); feats=echo_features(echo,lr)
                final_res,params,target,sl=predict_and_targets(net,a.variant,feats,pred.detach(),gt,ridge,lo,hi)
                final=(base.detach()+final_res).clamp(0,1); grad=.2*gradient_char(final,hr); loss=sl+grad
                (loss/accum).backward()
                if step%accum==0 or step==len(tr): opt.step(); opt.zero_grad(set_to_none=True)
                n=len(lr); sums+=np.array([sl.item(),grad.item(),loss.item()])*n; count+=n
            vm=validate(net,a.variant,edsr,residual,echo,va,device,ridge,lo,hi)
            row={"epoch":epoch,"train_scale_loss":sums[0]/count,"train_gradient_term":sums[1]/count,"train_total":sums[2]/count,**vm}; w.writerow(row); f.flush()
            print(f"{a.variant} {epoch:03d}/{a.epochs} train={row['train_total']:.6f} val={vm['val_total']:.6f} scale={vm['val_scale_loss']:.6f} grad={vm['val_gradient_term']:.6f}",flush=True)
            state={"epoch":epoch,"model":net.state_dict(),"optimizer":opt.state_dict(),"variant":a.variant,"val":vm,"ridge_config":cfg}
            torch.save(state,a.output/"last.pt")
            if vm["val_total"]<best: best=vm["val_total"]; best_epoch=epoch; torch.save(state,a.output/"best.pt")
            if vm["val_total"]<early_best-1e-4: early_best=vm["val_total"]; early_epoch=epoch
            if epoch-early_epoch>=50:
                print(f"EARLY_STOP patience=50 min_delta=1e-4 checkpoint_best_epoch={best_epoch} patience_reference_epoch={early_epoch} stop_epoch={epoch}",flush=True); break
    chosen=torch.load(a.output/"best.pt",map_location=device,weights_only=False); net.load_state_dict(chosen["model"]); net.eval()
    rows=[]; panels=[]; pred_dir=a.output/"test_predictions"; pred_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for lr,hr,names in te:
            lr,hr=lr.to(device),hr.to(device); base,pred,gt=frozen(lr,hr,edsr,residual)
            final_res,params,target,_=predict_and_targets(net,a.variant,echo_features(echo,lr),pred,gt,ridge,lo,hi)
            out=(base+final_res).clamp(0,1)[0,0].cpu().numpy(); hn=hr[0,0].cpu().numpy(); name=names[0]; roi=tumor_mask(PAIRED,name)
            rows.append({"method":a.variant,"file":name,**metric(out,hn,roi),"scale_mean":float(params[:,0:1].mean()),"scale_roi":0.0,"scale_nonroi":0.0})
            Image.fromarray((out*255).round().astype(np.uint8)).save(pred_dir/f"{name}_pred.png")
            canvas=Image.new("L",(768,292),255); draw=ImageDraw.Draw(canvas)
            for i,(im,title) in enumerate(((lr[0,0].cpu().numpy(),"LR"),(out,a.variant),(hn,"GT HR"))):
                canvas.paste(Image.fromarray((np.clip(im,0,1)*255).astype(np.uint8)),(i*256,36)); draw.text((i*256+8,10),title,fill=0)
            draw.text((8,274),name,fill=0); canvas.save(pred_dir/f"{name}_comparison.png"); panels.append(canvas)
    sheet=Image.new("L",(768,292*len(panels)),255)
    for i,p in enumerate(panels): sheet.paste(p,(0,292*i))
    sheet.save(a.output/"test_comparison_all.png")
    summary={"model":a.variant,"best_epoch":chosen["epoch"],"methods":[summarize(rows,a.variant)]}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2))
    with (a.output/"test_metrics_per_image.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with (a.output/"test_metrics_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        rec=summary["methods"][0]; w=csv.DictWriter(f,fieldnames=rec); w.writeheader(); w.writerow(rec)
    config={"variant":a.variant,"baseline":"ver7 Scale_Gradient_Lambda02","frozen":["Beta2 ROI upstream SR","Ver11-A residual predictor","EchoCare encoder"],"loss":"SmoothL1 scale supervision + 0.2 Gradient-Charbonnier","optimizer":"Adam","learning_rate":a.lr,"epochs":a.epochs,"seed":a.seed,"ridge":cfg}
    (a.output/"architecture_and_training_config.json").write_text(json.dumps(config,indent=2)); plot(a.output/"losses_and_scale_diagnostics.csv",a.output)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
