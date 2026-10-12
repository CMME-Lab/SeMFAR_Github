import argparse
import csv
import json
import math
import itertools
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ResidualBlock(nn.Module):
    def __init__(self, channels=64, scale=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, channels, 3, padding=1))
        self.scale = scale

    def forward(self, x):
        return x + self.body(x) * self.scale


class EDSR(nn.Module):
    def __init__(self, residual_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(1, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(scale=residual_scale) for _ in range(8)])
        self.body_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.tail = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        feature = self.head(x)
        feature = feature + self.body_conv(self.blocks(feature))  # global feature skip
        return x + self.tail(feature)  # image residual addition


def load_gray(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class PairedDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records, self.augment = records, augment

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        name, lr_path, hr_path = self.records[index]
        lr, hr = load_gray(lr_path), load_gray(hr_path)
        if self.augment:
            k, flip = random.randrange(4), random.random() < .5
            lr, hr = torch.rot90(lr, k, (1, 2)), torch.rot90(hr, k, (1, 2))
            if flip: lr, hr = lr.flip(2), hr.flip(2)
        return lr, hr, name


class UnpairedDataset(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, index): return load_gray(self.files[index])


def views(x):
    # The same geometry is mandatory for pixel-wise teacher/student consistency.
    k, flip = random.randrange(4), random.random() < .5
    weak = torch.rot90(x, k, (2, 3))
    if flip: weak = weak.flip(3)
    strong = weak.clone()
    contrast = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(0.8, 1.2)
    brightness = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(-0.08, 0.08)
    strong = ((strong - .5) * contrast + .5 + brightness + torch.randn_like(strong) * .03).clamp(0, 1)
    if random.random() < .5:
        size = random.randint(16, 48); y = random.randint(0, x.shape[-2] - size); z = random.randint(0, x.shape[-1] - size)
        strong[:, :, y:y+size, z:z+size] = strong.mean(dim=(2, 3), keepdim=True)
    return weak, strong


def charbonnier(a, b, eps=1e-3): return torch.sqrt((a - b).square() + eps * eps).mean()


GRADIENT_WEIGHT = 0.2


def gradient_charbonnier(pred, hr):
    pred_dx, hr_dx = pred[..., :, 1:] - pred[..., :, :-1], hr[..., :, 1:] - hr[..., :, :-1]
    pred_dy, hr_dy = pred[..., 1:, :] - pred[..., :-1, :], hr[..., 1:, :] - hr[..., :-1, :]
    return 0.5 * (charbonnier(pred_dx, hr_dx) + charbonnier(pred_dy, hr_dy))


def ssim(a, b):
    # Differentiable 11x11-window SSIM, averaged over batch and pixels.
    pool = nn.functional.avg_pool2d
    mu_a, mu_b = pool(a, 11, 1, 5), pool(b, 11, 1, 5)
    va = pool(a * a, 11, 1, 5) - mu_a.square()
    vb = pool(b * b, 11, 1, 5) - mu_b.square()
    vab = pool(a * b, 11, 1, 5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + .01**2) * (2 * vab + .03**2)) / ((mu_a.square()+mu_b.square()+.01**2)*(va+vb+.03**2))
    return score.mean()


@torch.no_grad()
def update_ema(teacher, student, mu):
    for t, s in zip(teacher.parameters(), student.parameters()): t.mul_(mu).add_(s, alpha=1-mu)
    for t, s in zip(teacher.buffers(), student.buffers()): t.copy_(s)


def discover(paired):
    records = []
    for lr in sorted(paired.glob("*_LR.png")):
        hr = lr.with_name(lr.name.replace("_LR.png", "_HR.png"))
        if hr.exists(): records.append((lr.stem[:-3], lr, hr))
    return records


@torch.no_grad()
def evaluate(model, loader, device, save_dir=None):
    model.eval(); sums = dict(char=0., ssim=0., grad_raw=0., grad_term=0., sup=0., psnr=0.); count = 0
    if save_dir: save_dir.mkdir(parents=True, exist_ok=True)
    for lr, hr, names in loader:
        lr, hr = lr.to(device), hr.to(device); pred = model(lr).clamp(0, 1)
        c, s = charbonnier(pred, hr), ssim(pred, hr)
        grad_raw = gradient_charbonnier(pred, hr); grad_term = GRADIENT_WEIGHT * grad_raw
        mse = (pred - hr).square().flatten(1).mean(1)
        n = len(lr); count += n
        sums['char'] += c.item()*n; sums['ssim'] += s.item()*n
        sums['grad_raw'] += grad_raw.item()*n; sums['grad_term'] += grad_term.item()*n
        sums['sup'] += (c+grad_term).item()*n
        sums['psnr'] += (-10*torch.log10(mse.clamp_min(1e-12))).sum().item()
        if save_dir:
            for i, name in enumerate(names):
                Image.fromarray((pred[i,0].cpu().numpy()*255).round().astype(np.uint8)).save(save_dir/f"{name}_pred.png")
    return {k: v/count for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paired', type=Path, default=Path('Paired')); ap.add_argument('--unpaired', type=Path, default=Path('Unpaired'))
    ap.add_argument('--split', type=Path)
    ap.add_argument('--output', type=Path, default=Path('outputs')); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=8); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--lambda-u', type=float, default=1.0, help='Maximum consistency weight')
    ap.add_argument('--lambda-l1', type=float, default=1.0)
    ap.add_argument('--lambda-u-warmup-epochs', type=int, default=0,
                    help='Keep lambda_u at zero through this epoch')
    ap.add_argument('--lambda-u-ramp-end-epoch', type=int, default=0,
                    help='Linearly reach --lambda-u at this epoch; 0 disables scheduling')
    ap.add_argument('--mu', type=float, default=.99); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-size', type=int, default=10)
    ap.add_argument('--test-size', type=int, default=3)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--max-unpaired-batches-per-epoch', type=int, default=0,
                    help='0 uses every unpaired image; otherwise uses this many shuffled batches per epoch')
    ap.add_argument('--cyclic-unpaired', action='store_true',
                    help='continue through one shuffled permutation across epoch boundaries')
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    records = discover(args.paired)
    if len(records) != 50: raise ValueError(f"Expected 50 complete pairs, found {len(records)}")
    if args.split:
        fixed = json.loads(args.split.read_text(encoding='utf-8-sig'))
        by_name = {item[0]: item for item in records}
        train = [by_name[name] for name in fixed['train']]
        val = [by_name[name] for name in fixed['validation']]
        test = [by_name[name] for name in fixed['test']]
    else:
        if args.val_size < 1 or args.test_size < 1 or args.val_size + args.test_size >= len(records):
            raise ValueError('val-size and test-size must be positive and leave at least one training pair')
        rng = random.Random(args.seed); rng.shuffle(records)
        test = records[:args.test_size]
        val = records[args.test_size:args.test_size + args.val_size]
        train = records[args.test_size + args.val_size:]
    unpaired = sorted(args.unpaired.glob("*.png"))
    if not unpaired: raise ValueError(f"No PNG images in {args.unpaired}")
    split = {'seed': args.seed, 'train': [x[0] for x in train], 'validation': [x[0] for x in val], 'test': [x[0] for x in test]}
    (args.output/'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student, teacher = EDSR().to(device), EDSR().to(device); teacher.load_state_dict(student.state_dict())
    teacher.requires_grad_(False); teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    train_loader = DataLoader(PairedDataset(train, True), args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type=='cuda')
    unpaired_loader = DataLoader(UnpairedDataset(unpaired), args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type=='cuda')
    val_loader = DataLoader(PairedDataset(val), args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(PairedDataset(test), 1, num_workers=args.num_workers)
    fields = ['epoch','lambda_u','train_char','train_gradient_raw','train_gradient_term','train_sup','train_cons_char','train_cons_l1','train_cons','train_total','val_char','val_ssim','val_gradient_raw','val_gradient_term','val_sup','val_psnr']
    best = math.inf; best_epoch = 0; early_best = math.inf; early_epoch = 0
    unpaired_iterator = [iter(unpaired_loader)]
    with (args.output/'losses.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for epoch in range(1, args.epochs+1):
            if not args.lambda_u_ramp_end_epoch:
                current_lambda_u = args.lambda_u
            elif epoch <= args.lambda_u_warmup_epochs + 1:
                current_lambda_u = 0.0
            elif epoch >= args.lambda_u_ramp_end_epoch:
                current_lambda_u = args.lambda_u
            else:
                denominator = args.lambda_u_ramp_end_epoch - args.lambda_u_warmup_epochs - 1
                current_lambda_u = args.lambda_u * (epoch - args.lambda_u_warmup_epochs - 1) / denominator
            student.train(); sums = np.zeros(9); steps = 0; paired_iter = iter(train_loader)
            count = args.max_unpaired_batches_per_epoch or len(unpaired_loader)
            if args.cyclic_unpaired:
                epoch_batches=[]
                for _ in range(count):
                    try: u=next(unpaired_iterator[0])
                    except StopIteration: unpaired_iterator[0]=iter(unpaired_loader); u=next(unpaired_iterator[0])
                    epoch_batches.append(u)
            else: epoch_batches=itertools.islice(unpaired_loader,count)
            for u in epoch_batches:
                try: lr, hr, _ = next(paired_iter)
                except StopIteration: paired_iter = iter(train_loader); lr, hr, _ = next(paired_iter)
                lr, hr, u = lr.to(device), hr.to(device), u.to(device); weak, strong = views(u)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                    pred = student(lr); c = charbonnier(pred, hr)
                    grad_raw = gradient_charbonnier(pred, hr); grad_term = GRADIENT_WEIGHT * grad_raw; sup = c+grad_term
                    with torch.no_grad(): pseudo = teacher(weak)
                    strong_pred = student(strong); cc = charbonnier(strong_pred, pseudo); l1 = nn.functional.l1_loss(strong_pred, pseudo)
                    cons = cc + args.lambda_l1*l1; total = sup + current_lambda_u*cons
                scaler.scale(total).backward(); scaler.step(optimizer); scaler.update(); update_ema(teacher, student, args.mu)
                sums += [c.item(),grad_raw.item(),grad_term.item(),sup.item(),cc.item(),l1.item(),cons.item(),total.item(),1]; steps += 1
            avg = sums[:8]/sums[8]; vm = evaluate(student, val_loader, device)
            row = dict(epoch=epoch, lambda_u=current_lambda_u, train_char=avg[0], train_gradient_raw=avg[1], train_gradient_term=avg[2],
                       train_sup=avg[3], train_cons_char=avg[4], train_cons_l1=avg[5], train_cons=avg[6],
                       train_total=avg[7], val_char=vm['char'], val_ssim=vm['ssim'], val_gradient_raw=vm['grad_raw'],
                       val_gradient_term=vm['grad_term'], val_sup=vm['sup'], val_psnr=vm['psnr'])
            writer.writerow(row); f.flush()
            print(f"Epoch {epoch:03d}/{args.epochs} lambda_u={current_lambda_u:.4f} total={avg[7]:.6f} sup={avg[3]:.6f} [char={avg[0]:.6f}, grad02={avg[2]:.6f}] cons={avg[6]:.6f} [char={avg[4]:.6f}, l1={avg[5]:.6f}] val_sup={vm['sup']:.6f} val_grad02={vm['grad_term']:.6f} val_ssim={vm['ssim']:.4f} val_psnr={vm['psnr']:.2f}", flush=True)
            state = {'epoch':epoch, 'student':student.state_dict(), 'teacher':teacher.state_dict(), 'optimizer':optimizer.state_dict(), 'args':vars(args)}
            torch.save(state, args.output/'last.pt')
            if vm['sup'] < best: best=vm['sup']; best_epoch=epoch; torch.save(state, args.output/'best.pt')
            if vm['sup']<early_best-1e-4: early_best=vm['sup']; early_epoch=epoch
            if epoch-early_epoch>=50:
                print(f'EARLY_STOP patience=50 min_delta=1e-4 checkpoint_best_epoch={best_epoch} patience_reference_epoch={early_epoch} stop_epoch={epoch}',flush=True); break
    checkpoint = torch.load(args.output/'best.pt', map_location=device, weights_only=False); student.load_state_dict(checkpoint['student'])
    metrics = evaluate(student, test_loader, device, args.output/'test_predictions')
    metrics['best_epoch'] = checkpoint['epoch']; metrics['test_files'] = [x[0] for x in test]
    (args.output/'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('TEST', json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()


