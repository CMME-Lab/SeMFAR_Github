import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1))

    def forward(self, x):
        return x + 0.1 * self.body(x)


class EDSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(1, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock() for _ in range(8)])
        self.body_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.tail = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        head = self.head(x)
        features = head + self.body_conv(self.blocks(head))
        return x + self.tail(features)


def load_gray(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class PairedDataset(Dataset):
    def __init__(self, paired_dir, names, augment=False):
        self.paired_dir, self.names, self.augment = paired_dir, names, augment

    def __len__(self): return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        lr = load_gray(self.paired_dir / f'{name}_LR.png')
        hr = load_gray(self.paired_dir / f'{name}_HR.png')
        if self.augment:
            lr, hr = paired_augment(lr, hr)
        return lr, hr, name


def paired_augment(lr, hr):
    """Apply exactly the same random geometry to an aligned LR-HR pair."""
    k = random.randrange(4)
    horizontal_flip = random.random() < 0.5
    lr = torch.rot90(lr, k, (1, 2))
    hr = torch.rot90(hr, k, (1, 2))
    if horizontal_flip:
        lr = lr.flip(2)
        hr = hr.flip(2)
    return lr, hr


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b).square() + eps * eps).mean()


def ssim(a, b):
    pool = nn.functional.avg_pool2d
    ma, mb = pool(a, 11, 1, 5), pool(b, 11, 1, 5)
    va, vb = pool(a*a, 11, 1, 5)-ma.square(), pool(b*b, 11, 1, 5)-mb.square()
    vab = pool(a*b, 11, 1, 5)-ma*mb
    return (((2*ma*mb+.01**2)*(2*vab+.03**2)) /
            ((ma.square()+mb.square()+.01**2)*(va+vb+.03**2))).mean()


def gradient_charbonnier(pred, hr):
    pred_dx, hr_dx = pred[..., :, 1:] - pred[..., :, :-1], hr[..., :, 1:] - hr[..., :, :-1]
    pred_dy, hr_dy = pred[..., 1:, :] - pred[..., :-1, :], hr[..., 1:, :] - hr[..., :-1, :]
    return 0.5 * (charbonnier(pred_dx, hr_dx) + charbonnier(pred_dy, hr_dy))


def supervised_terms(pred, hr):
    char = charbonnier(pred, hr)
    score = ssim(pred, hr)
    grad_raw = gradient_charbonnier(pred, hr)
    grad_term = 0.2 * grad_raw
    return char, score, grad_raw, grad_term, char + grad_term


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); totals = np.zeros(6); count = 0
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr).clamp(0, 1)
        char, score, grad_raw, grad_term, loss = supervised_terms(pred, hr)
        psnr = (-10 * torch.log10((pred-hr).square().flatten(1).mean(1).clamp_min(1e-12))).sum()
        n = len(lr); totals += [char.item()*n, score.item()*n, grad_raw.item()*n, grad_term.item()*n, loss.item()*n, psnr.item()]; count += n
    return dict(char=totals[0]/count, ssim=totals[1]/count, gradient_raw=totals[2]/count,
                gradient_term=totals[3]/count, loss=totals[4]/count, psnr=totals[5]/count)


@torch.no_grad()
def test_and_save(model, loader, device, output):
    model.eval(); rows, panels = [], []
    pred_dir = output / 'test_predictions'; pred_dir.mkdir(parents=True, exist_ok=True)
    for lr, hr, names in loader:
        name = names[0]; lr, hr = lr.to(device), hr.to(device); pred = model(lr).clamp(0, 1)
        char, pred_ssim, grad_raw, grad_term, loss = supervised_terms(pred, hr)
        pred_psnr = (-10*torch.log10((pred-hr).square().mean().clamp_min(1e-12))).item()
        lr_ssim = ssim(lr, hr).item(); lr_psnr = (-10*torch.log10((lr-hr).square().mean().clamp_min(1e-12))).item()
        row = dict(file=name, charbonnier=char.item(), gradient_raw=grad_raw.item(),
                   gradient_term=grad_term.item(), supervised_loss=loss.item(),
                   prediction_ssim=pred_ssim.item(), prediction_psnr_db=pred_psnr,
                   lr_input_ssim=lr_ssim, lr_input_psnr_db=lr_psnr,
                   ssim_improvement=pred_ssim.item()-lr_ssim, psnr_improvement_db=pred_psnr-lr_psnr)
        rows.append(row)
        arrays = [(x[0,0].cpu().numpy()*255).round().astype(np.uint8) for x in (lr,pred,hr)]
        Image.fromarray(arrays[1]).save(pred_dir/f'{name}_pred.png')
        panel = Image.new('L', (768,292), 255); draw = ImageDraw.Draw(panel)
        for i,(array,label) in enumerate(zip(arrays,('LR input','Prediction','HR ground truth'))):
            panel.paste(Image.fromarray(array),(i*256,36)); draw.text((i*256+8,10),label,fill=0)
        draw.text((8,274),f"{name} | Pred PSNR {pred_psnr:.2f} dB | Pred SSIM {pred_ssim.item():.4f}",fill=0)
        panel.save(pred_dir/f'{name}_comparison.png'); panels.append(panel)
        print(f"TEST {name}: loss={loss.item():.6f} char={char.item():.6f} grad02={grad_term.item():.6f} SSIM={pred_ssim.item():.4f} PSNR={pred_psnr:.2f}", flush=True)
    with (output/'test_per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    sheet=Image.new('L',(768,292*len(panels)),255)
    for i,panel in enumerate(panels): sheet.paste(panel,(0,292*i))
    sheet.save(output/'test_comparison_all.png')
    summary={'num_test_images':len(rows)}
    for key in rows[0]:
        if key=='file': continue
        values=np.asarray([r[key] for r in rows]); summary[key]={'mean':float(values.mean()),'std':float(values.std(ddof=1))}
    (output/'test_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--paired',type=Path,default=Path('../Paired'))
    ap.add_argument('--split',type=Path,default=Path('../EMA_teacher/outputs_2/split.json'))
    ap.add_argument('--output',type=Path,default=Path('outputs'))
    ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=0.01)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    split=json.loads(args.split.read_text(encoding='utf-8-sig'))
    if [len(split[k]) for k in ('train','validation','test')] != [30,10,10]: raise ValueError('Split must be 30/10/10')
    (args.output/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=EDSR().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    train_loader=DataLoader(PairedDataset(args.paired,split['train'],True),args.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    val_loader=DataLoader(PairedDataset(args.paired,split['validation']),args.batch_size,num_workers=0)
    test_loader=DataLoader(PairedDataset(args.paired,split['test']),1,num_workers=0)
    fields=['epoch','train_char','train_ssim','train_gradient_raw','train_gradient_term','train_loss','train_psnr',
            'val_char','val_ssim','val_gradient_raw','val_gradient_term','val_loss','val_psnr']
    best=float('inf'); best_epoch=0; early_best=float('inf'); early_epoch=0
    with (args.output/'losses.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for epoch in range(1,args.epochs+1):
            model.train(); totals=np.zeros(6); count=0
            for lr,hr,_ in train_loader:
                lr,hr=lr.to(device),hr.to(device); optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                    pred=model(lr); char,score,grad_raw,grad_term,loss=supervised_terms(pred,hr)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                with torch.no_grad(): psnr=(-10*torch.log10((pred-hr).square().flatten(1).mean(1).clamp_min(1e-12))).sum()
                n=len(lr); totals += [char.item()*n,score.item()*n,grad_raw.item()*n,grad_term.item()*n,loss.item()*n,psnr.item()]; count+=n
            tm=totals/count; vm=evaluate(model,val_loader,device)
            row=dict(zip(fields,[epoch,*tm,vm['char'],vm['ssim'],vm['gradient_raw'],vm['gradient_term'],vm['loss'],vm['psnr']]))
            writer.writerow(row); f.flush()
            print(f"Epoch {epoch:03d}/{args.epochs} TRAIN loss={tm[4]:.6f} [char={tm[0]:.6f}, grad02={tm[3]:.6f}] SSIM={tm[1]:.4f} PSNR={tm[5]:.2f} | VALID loss={vm['loss']:.6f} [char={vm['char']:.6f}, grad02={vm['gradient_term']:.6f}] SSIM={vm['ssim']:.4f} PSNR={vm['psnr']:.2f}",flush=True)
            state={'epoch':epoch,'student':model.state_dict(),'optimizer':optimizer.state_dict(),'args':vars(args)}
            torch.save(state,args.output/'last.pt')
            if vm['loss']<best: best=vm['loss']; best_epoch=epoch; torch.save(state,args.output/'best.pt')
            if vm['loss']<early_best-1e-4: early_best=vm['loss']; early_epoch=epoch
            if epoch-early_epoch>=50:
                print(f'EARLY_STOP patience=50 min_delta=1e-4 checkpoint_best_epoch={best_epoch} patience_reference_epoch={early_epoch} stop_epoch={epoch}',flush=True); break
    ckpt=torch.load(args.output/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ckpt['student'])
    summary=test_and_save(model,test_loader,device,args.output); summary['best_epoch']=ckpt['epoch']
    (args.output/'test_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    p=summary['prediction_psnr_db']; s=summary['prediction_ssim']
    (args.output/'architecture_and_training_config.json').write_text(json.dumps({
        'model':'SeMFAR upstream EDSR-lite (8 residual blocks, residual scaling 0.1)',
        'loss':'Charbonnier reconstruction + 0.2*gradient-Charbonnier',
        'gradient':'0.5*(Charbonnier(horizontal finite differences)+Charbonnier(vertical finite differences))',
        'optimizer':'Adam','learning_rate':args.lr,'weight_decay':0.0,
        'split':'30 train / 10 validation / 10 test','epochs':args.epochs,'seed':args.seed
    },indent=2),encoding='utf-8')
    print(f"TEST SUMMARY (n=10): loss={summary['supervised_loss']['mean']:.6f} PSNR={p['mean']:.4f}+/-{p['std']:.4f} SSIM={s['mean']:.4f}+/-{s['std']:.4f}",flush=True)


if __name__=='__main__': main()


