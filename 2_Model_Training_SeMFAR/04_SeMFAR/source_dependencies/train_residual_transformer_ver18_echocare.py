"""Ver18: Ver11-A residual Transformer with frozen EchoCare F3 conditioning."""
import argparse, csv, json, random, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from monai.networks.nets.swin_unetr import SwinTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIT = ROOT/"DiT"
DRT = ROOT/"Deterministic_Residual_Transformer"
sys.path[:0] = [str(DIT), str(DRT)]
from train_residual_dit import EDSR_ROOT, EDSR, Pairs, discover, evaluate_images, fixed_2d_position
from train_residual_dit_ver4 import charbonnier
from train_residual_dit_ver5 import pooled_corr
from train_residual_transformer_ver7 import TransformerBlock
from train_residual_transformer_ver11 import gaussian

ECHO_CHECKPOINT = HERE/"EchoCare"/"checkpoints"/"echocare_encoder.pth"


class EchoConditionedResidualTransformer(nn.Module):
    def __init__(self, dim=256, depth=8, heads=8, patch=8):
        super().__init__(); self.patch = patch
        self.lr_conv = nn.Conv2d(1,64,3,padding=1); self.coarse_conv = nn.Conv2d(1,64,3,padding=1)
        self.lr_patch = nn.Conv2d(64,dim,patch,stride=patch); self.coarse_patch = nn.Conv2d(64,dim,patch,stride=patch)
        self.q_norm = nn.LayerNorm(dim); self.c_norm = nn.LayerNorm(dim)
        self.coarse_cross = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.echo_proj = nn.Conv2d(512,dim,1); self.echo_q_norm = nn.LayerNorm(dim); self.echo_kv_norm = nn.LayerNorm(dim)
        self.echo_cross = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.register_buffer("position",fixed_2d_position(256//patch,dim),persistent=True)
        self.blocks = nn.ModuleList([TransformerBlock(dim,heads,4) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.out = nn.Linear(dim,patch*patch)

    def forward(self, lr, coarse, echo_f3):
        q = self.lr_patch(self.lr_conv(lr)).flatten(2).transpose(1,2)
        c = self.coarse_patch(self.coarse_conv(coarse)).flatten(2).transpose(1,2)
        x = self.coarse_cross(self.q_norm(q),self.c_norm(c),self.c_norm(c),need_weights=False)[0]
        echo = self.echo_proj(echo_f3).flatten(2).transpose(1,2)
        echo = self.echo_kv_norm(echo)
        x = x + self.echo_cross(self.echo_q_norm(x),echo,echo,need_weights=False)[0]
        x = x + self.position[None].to(x.dtype)
        for block in self.blocks: x = block(x)
        p=self.patch; b=x.shape[0]
        return self.out(self.norm(x)).view(b,32,32,p,p).permute(0,1,3,2,4).reshape(b,1,256,256)


def make_echocare(device):
    encoder = SwinTransformer(in_chans=3,embed_dim=128,window_size=[8]*2,patch_size=[2]*2,
                              depths=[2,2,18,2],num_heads=[4,8,16,32],mlp_ratio=4.0,qkv_bias=True,
                              use_checkpoint=False,spatial_dims=2,use_v2=True)
    state = torch.load(ECHO_CHECKPOINT,map_location="cpu",weights_only=False)
    state.pop("mask_token",None); encoder.load_state_dict(state,strict=True)
    return encoder.to(device).eval().requires_grad_(False)


@torch.no_grad()
def echo_f3(encoder, lr):
    return encoder(lr.repeat(1,3,1,1))[2]


@torch.no_grad()
def validate(model, edsr, echo, loader, device, eps, sigma, lambda_full):
    model.eval(); sums=np.zeros(3); count=0; pred_low=[]; gt_low=[]; pred_full=[]; gt_full=[]
    for lr,hr,_ in loader:
        lr,hr=lr.to(device),hr.to(device); coarse=edsr(lr); target=hr-coarse; features=echo_f3(echo,lr)
        pred=model(lr,coarse,features); yl=gaussian(target,sigma); pl=gaussian(pred,sigma)
        low=charbonnier(pl,yl,eps); full=charbonnier(pred,target,eps); total=low+lambda_full*full
        sums += np.array([total.item(),low.item(),full.item()])*len(lr); count += len(lr)
        pred_low.append(pl.cpu().flatten()); gt_low.append(yl.cpu().flatten()); pred_full.append(pred.cpu().flatten()); gt_full.append(target.cpu().flatten())
    pl,yl,pf,yf=map(torch.cat,(pred_low,gt_low,pred_full,gt_full))
    return {"val_total":sums[0]/count,"val_low_charbonnier":sums[1]/count,"val_full_charbonnier":sums[2]/count,
            "val_low_mse":float(F.mse_loss(pl,yl)),"val_low_corr":pooled_corr(pl,yl),"val_low_pred_std":float(pl.std(unbiased=False)),
            "val_low_gt_std":float(yl.std(unbiased=False)),"val_low_std_ratio":float(pl.std(unbiased=False)/yl.std(unbiased=False).clamp_min(1e-12)),
            "val_full_mse":float(F.mse_loss(pf,yf)),"val_full_corr":pooled_corr(pf,yf)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--paired",type=Path,default=ROOT/"Paired")
    ap.add_argument("--split",type=Path,default=EDSR_ROOT/"Comparison_300epochs_CyclicUnpaired_4064"/"Difficulty_Ver1"/"split.json")
    ap.add_argument("--edsr-checkpoint",type=Path,default=EDSR_ROOT/"Comparison_300epochs_CyclicUnpaired_4064"/"Difficulty_Ver1"/"best.pt")
    ap.add_argument("--output",type=Path,default=HERE/"Ver18_EchoCare_Residual_Transformer_300epochs")
    ap.add_argument("--epochs",type=int,default=300); ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--effective-batch-size",type=int,default=16); ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--sigma",type=float,default=1.5)
    ap.add_argument("--lambda-full",type=float,default=.1); ap.add_argument("--eta-low",type=float,default=1.0)
    ap.add_argument("--eta-high",type=float,default=0.0); ap.add_argument("--charbonnier-eps",type=float,default=1e-3)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.effective_batch_size % args.batch_size: raise ValueError("effective batch size must be divisible by batch size")
    accum=args.effective_batch_size//args.batch_size
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    split=json.loads(args.split.read_text()); records={r[0]:r for r in discover(args.paired)}
    sets={k:[records[n] for n in split[k]] for k in ("train","validation","test")}; (args.output/"split.json").write_text(json.dumps(split,indent=2))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(args.edsr_checkpoint,map_location=device,weights_only=False); edsr=EDSR(ck.get("args",{}).get("residual_scale",.1)).to(device)
    edsr.load_state_dict(ck["student"]); edsr.eval().requires_grad_(False); echo=make_echocare(device)
    model=EchoConditionedResidualTransformer().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    train=DataLoader(Pairs(sets["train"],True),args.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=="cuda")
    val=DataLoader(Pairs(sets["validation"]),args.batch_size); test=DataLoader(Pairs(sets["test"]),1)
    val_fields=["val_total","val_low_charbonnier","val_full_charbonnier","val_low_mse","val_low_corr","val_low_pred_std","val_low_gt_std","val_low_std_ratio","val_full_mse","val_full_corr"]
    fields=["epoch","train_total","train_low_charbonnier","train_full_charbonnier",*val_fields]; best=float("inf")
    with (args.output/"losses_and_validation_diagnostics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for epoch in range(1,args.epochs+1):
            model.train(); sums=np.zeros(3); count=0; optimizer.zero_grad(set_to_none=True)
            for step,(lr,hr,_) in enumerate(train,1):
                lr,hr=lr.to(device),hr.to(device)
                with torch.no_grad(): coarse=edsr(lr); target=hr-coarse; yl=gaussian(target,args.sigma); features=echo_f3(echo,lr)
                with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
                    pred=model(lr,coarse,features); low=charbonnier(gaussian(pred,args.sigma),yl,args.charbonnier_eps)
                    full=charbonnier(pred,target,args.charbonnier_eps); loss=low+args.lambda_full*full
                scaler.scale(loss/accum).backward()
                if step%accum==0 or step==len(train): scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                sums += np.array([loss.item(),low.item(),full.item()])*len(lr); count += len(lr)
            stats=validate(model,edsr,echo,val,device,args.charbonnier_eps,args.sigma,args.lambda_full); train_stats=sums/count
            row=dict(zip(fields,[epoch,*train_stats,*stats.values()])); writer.writerow(row); handle.flush()
            print(f'Ver18 {epoch:03d}/{args.epochs} train={train_stats[0]:.6f} low={train_stats[1]:.6f} full={train_stats[2]:.6f} | val={stats["val_total"]:.6f} low-corr={stats["val_low_corr"]:.4f} ratio={stats["val_low_std_ratio"]:.3f} full-corr={stats["val_full_corr"]:.4f}',flush=True)
            state={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"args":vars(args),"edsr_checkpoint":str(args.edsr_checkpoint),"echo_checkpoint":str(ECHO_CHECKPOINT),"val_total":stats["val_total"]}
            torch.save(state,args.output/"last.pt")
            if stats["val_total"]<best: best=stats["val_total"]; torch.save(state,args.output/"best.pt")

    state=torch.load(args.output/"best.pt",map_location=device,weights_only=False); model.load_state_dict(state["model"]); model.eval()
    pred_dir=args.output/"test_predictions"; pred_dir.mkdir(exist_ok=True); diag=args.output/"residual_diagnostics"; diag.mkdir(exist_ok=True)
    panels=[]; residual_panels=[]; rows=[]; all_pl=[]; all_yl=[]; all_p=[]; all_y=[]
    with torch.no_grad():
        for lr,hr,names in test:
            lr,hr=lr.to(device),hr.to(device); coarse=edsr(lr); target=hr-coarse; pred=model(lr,coarse,echo_f3(echo,lr))
            pl=gaussian(pred,args.sigma); yl=gaussian(target,args.sigma); applied=args.eta_low*pl+args.eta_high*(pred-pl); final=(coarse+applied).clamp(0,1)
            name=names[0]; yn=target[0,0].cpu().numpy(); yln=yl[0,0].cpu().numpy(); pn=pred[0,0].cpu().numpy(); pln=pl[0,0].cpu().numpy()
            all_pl.append(pln.ravel());all_yl.append(yln.ravel());all_p.append(pn.ravel());all_y.append(yn.ravel())
            rows.append({"file":name,"low_mse":float(np.mean((pln-yln)**2)),"low_mae":float(np.mean(np.abs(pln-yln))),"low_corr":pooled_corr(torch.from_numpy(pln),torch.from_numpy(yln)),"low_std_pred":float(pln.std()),"low_std_gt":float(yln.std()),"low_std_ratio":float(pln.std()/yln.std()),"full_mse":float(np.mean((pn-yn)**2)),"full_corr":pooled_corr(torch.from_numpy(pn),torch.from_numpy(yn))})
            arrays=[(z[0,0].cpu().numpy()*255).clip(0,255).round().astype(np.uint8) for z in (lr,coarse,final,hr)]
            Image.fromarray(arrays[2]).save(pred_dir/f"{name}_pred.png"); canvas=Image.new("L",(1024,292),255); draw=ImageDraw.Draw(canvas)
            for i,(image,label) in enumerate(zip(arrays,("LR input","Frozen Difficulty EDSR","EDSR + Ver18 EchoCare","HR ground truth"))): canvas.paste(Image.fromarray(image),(i*256,36));draw.text((i*256+8,10),label,fill=0)
            draw.text((8,274),name,fill=0);canvas.save(pred_dir/f"{name}_comparison.png");panels.append(canvas)
            signed=lambda x:np.clip((x/.8+.5)*255,0,255).round().astype(np.uint8); rc=Image.new("L",(1024,292),255); rd=ImageDraw.Draw(rc)
            for i,(image,label) in enumerate(zip((signed(yn),signed(yln),signed(pn),signed(pln)),("Full R_GT","Low R_GT","Raw R_hat","Low R_hat"))):rc.paste(Image.fromarray(image),(i*256,36));rd.text((i*256+8,10),label,fill=0)
            rd.text((8,274),f'{name} low corr={rows[-1]["low_corr"]:.4f} ratio={rows[-1]["low_std_ratio"]:.3f}',fill=0);rc.save(diag/f"{name}_residual_comparison.png");residual_panels.append(rc)
    sheet=Image.new("L",(1024,292*len(panels)),255); rsheet=Image.new("L",(1024,292*len(residual_panels)),255)
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*292))
    for i,panel in enumerate(residual_panels):rsheet.paste(panel,(0,i*292))
    sheet.save(args.output/"test_comparison_all.png");rsheet.save(diag/"test_residual_comparison_all.png")
    summary=evaluate_images(args.output,args.paired,split["test"],state["epoch"]); summary.update({"model":"Ver18 EchoCare-conditioned Ver11-A","sigma":args.sigma,"lambda_full":args.lambda_full,"eta_low":args.eta_low,"eta_high":args.eta_high,"checkpoint_metric":"minimum validation total loss"});(args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    pl,yl,p,y=map(np.concatenate,(all_pl,all_yl,all_p,all_y)); low_mse=float(np.mean((pl-yl)**2)); zero=float(np.mean(yl**2))
    residual_summary={"n_images":len(rows),"pooled":{"low_mse":low_mse,"low_mae":float(np.mean(np.abs(pl-yl))),"low_correlation":pooled_corr(torch.from_numpy(pl),torch.from_numpy(yl)),"low_std_pred":float(pl.std()),"low_std_gt":float(yl.std()),"low_std_ratio":float(pl.std()/yl.std()),"low_zero_mse":zero,"low_error_to_zero_ratio":low_mse/zero,"full_mse":float(np.mean((p-y)**2)),"full_correlation":pooled_corr(torch.from_numpy(p),torch.from_numpy(y)),"full_std_pred":float(p.std()),"full_std_gt":float(y.std())}}
    (diag/"residual_metrics_summary.json").write_text(json.dumps(residual_summary,indent=2))
    with (diag/"per_image_residual_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f: writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    history=list(csv.DictReader((args.output/"losses_and_validation_diagnostics.csv").open())); x=[int(r["epoch"]) for r in history];fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot(x,[float(r["train_total"]) for r in history],label="train");axes[0].plot(x,[float(r["val_total"]) for r in history],label="val");axes[1].plot(x,[float(r["val_low_corr"]) for r in history]);axes[2].plot(x,[float(r["val_low_std_ratio"]) for r in history])
    for ax,title in zip(axes,("Total loss","Validation low-residual correlation","Validation low-residual std ratio")):ax.set_title(title);ax.grid(alpha=.25);ax.set_xlabel("Epoch");ax.legend() if len(ax.lines)>1 else None
    fig.tight_layout();fig.savefig(args.output/"loss_and_residual_diagnostics.png",dpi=180);plt.close(fig)
    config={"architecture":"Ver11-A plus frozen EchoCare F3 cross-attention","echo_feature":"F3 [B,512,32,32] -> trainable 1x1 Conv 512->256 -> 1024 tokens","frozen":["Difficulty Ver1 EDSR","EchoCare Swin encoder"],"trainable":["EchoCare projector","EchoCare cross-attention","Residual Transformer"],"loss":f"Char(G(Rhat),G(RGT))+{args.lambda_full}*Char(Rhat,RGT)","inference":f"EDSR+{args.eta_low}*G(Rhat)+{args.eta_high}*(Rhat-G(Rhat))","epochs":args.epochs,"physical_batch":args.batch_size,"effective_batch":args.effective_batch_size}
    (args.output/"architecture_and_training_config.json").write_text(json.dumps(config,indent=2));(args.output/"train_residual_transformer_ver18_echocare_frozen.py").write_text(Path(__file__).read_text(encoding="utf-8"),encoding="utf-8")
    print(json.dumps({"test":summary,"residual":residual_summary},indent=2),flush=True)


if __name__=="__main__": main()
