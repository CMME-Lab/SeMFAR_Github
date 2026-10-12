"""Ver16-A/B: ridge-regularized 8x8 local-scale targets for frozen Ver11-A."""
import argparse, csv, json, random, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "DiT"), str(HERE)]
from train_residual_dit import EDSR, Pairs, discover, ssim_map, tumor_mask, otsu, dice
from train_residual_transformer_ver7 import DeterministicResidualTransformer
from train_residual_transformer_ver11 import gaussian
from train_reliability_gate_ver14 import PatchGate

V11 = HERE / "Residual_Transformer_Ver11_A_LambdaFull_0.1_300epochs"
ALPHA, EPS = 1.3876873254776, 1e-12


def components(gt, pred):
    numerator = F.avg_pool2d(gt * pred, 8, 8)
    energy = F.avg_pool2d(pred.square(), 8, 8)
    return numerator, energy


def ridge_target(gt, pred, ridge, lo, hi):
    numerator, energy = components(gt, pred)
    return (numerator / (energy + ridge)).clamp(lo, hi), energy


def corr(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False) + EPS))


def metric(p, h, roi):
    sm, th = ssim_map(p, h), otsu(h)
    return {"psnr": float(-10*np.log10(max(np.mean((p-h)**2), EPS))),
            "ssim": float(sm.mean()),
            "roi_psnr": float(-10*np.log10(max(np.mean((p[roi]-h[roi])**2), EPS))),
            "roi_ssim": float(sm[roi].mean()), "dice": dice(p < th, h < th)}


@torch.no_grad()
def frozen(lr, hr, edsr, residual):
    base = edsr(lr)
    pred = gaussian(residual(lr, base), 1.5)
    return base, pred, gaussian(hr-base, 1.5)


def pred_scale(net, base, pred, lo, hi):
    return lo + (hi-lo) * torch.sigmoid(net(base, pred))


def scale_loss(pred, target, energy, ridge, weighted):
    raw = F.smooth_l1_loss(pred, target, reduction="none")
    if not weighted:
        return raw.mean()
    weight = energy / (energy + ridge)
    return (weight * raw).sum() / (weight.sum() + EPS)


@torch.no_grad()
def validate(net, edsr, residual, loader, device, ridge, lo, hi, weighted):
    net.eval(); pp, yy, ww, losses = [], [], [], []
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        base, pred, gt = frozen(lr, hr, edsr, residual)
        target, energy = ridge_target(gt, pred, ridge, lo, hi)
        estimate = pred_scale(net, base, pred, lo, hi)
        losses.append(scale_loss(estimate, target, energy, ridge, weighted).item())
        pp.append(estimate.cpu()); yy.append(target.cpu()); ww.append((energy/(energy+ridge)).cpu())
    pp, yy, ww = map(torch.cat, (pp, yy, ww))
    return {"val_scale_loss": float(np.mean(losses)), "val_scale_mse": float(F.mse_loss(pp, yy)),
            "val_scale_mae": float(F.l1_loss(pp, yy)), "val_scale_corr": corr(pp, yy),
            "val_scale_mean": float(pp.mean()), "val_target_mean": float(yy.mean()),
            "val_weight_mean": float(ww.mean())}


def summarize(rows, name):
    selected = [r for r in rows if r["method"] == name]
    out = {"method": name, "n": len(selected)}
    for key in ("psnr", "ssim", "roi_psnr", "roi_ssim", "dice"):
        values = np.asarray([r[key] for r in selected])
        out[key+"_mean"], out[key+"_std"] = float(values.mean()), float(values.std(ddof=1))
    for key in ("scale_mean", "scale_roi", "scale_nonroi"):
        out[key] = float(np.mean([r[key] for r in selected]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", choices=("A", "B"), required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state11 = torch.load(V11/"best.pt", map_location=device, weights_only=False)
    split = json.loads((V11/"split.json").read_text())
    records = {x[0]: x for x in discover(ROOT/"Paired")}
    sets = {k: [records[n] for n in split[k]] for k in ("train", "validation", "test")}
    (args.output/"split.json").write_text(json.dumps(split, indent=2))
    checkpoint = torch.load(Path(state11["edsr_checkpoint"]), map_location=device, weights_only=False)
    edsr = EDSR(checkpoint.get("args", {}).get("residual_scale", .1)).to(device)
    edsr.load_state_dict(checkpoint["student"]); edsr.eval().requires_grad_(False)
    residual = DeterministicResidualTransformer().to(device)
    residual.load_state_dict(state11["model"]); residual.eval().requires_grad_(False)
    train = DataLoader(Pairs(sets["train"], True), args.batch_size, shuffle=True)
    val = DataLoader(Pairs(sets["validation"]), args.batch_size)
    test = DataLoader(Pairs(sets["test"]), 1)

    nums, energies = [], []
    with torch.no_grad():
        for lr, hr, _ in val:
            base, pred, gt = frozen(lr.to(device), hr.to(device), edsr, residual)
            num, energy = components(gt, pred); nums.append(num.cpu()); energies.append(energy.cpu())
    nums, energies = torch.cat(nums), torch.cat(energies)
    positive = energies[energies > 0]
    eq = torch.quantile(positive, torch.tensor([.01, .05, .10, .25, .50, .75, .95, .99]))
    ridge = float(max(eq[2], torch.tensor(1e-12)))
    raw = nums/(energies+ridge)
    rq = torch.quantile(raw.flatten(), torch.tensor([.01, .05, .25, .50, .75, .95, .99]))
    lo, hi = 0., float(min(3., max(.5, rq[5].item())))
    cfg = {"variant": args.variant, "ridge_rule": "validation positive residual-energy q10",
           "ridge_lambda": ridge, "energy_q01_q05_q10_q25_q50_q75_q95_q99": [float(x) for x in eq],
           "target_q01_q05_q25_q50_q75_q95_q99": [float(x) for x in rq], "scale_bounds": [lo, hi],
           "weighted_loss": args.variant == "B", "epochs": args.epochs}
    (args.output/"ridge_and_target_config.json").write_text(json.dumps(cfg, indent=2)); print(json.dumps(cfg, indent=2), flush=True)

    net, weighted = PatchGate().to(device), args.variant == "B"
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    fields = ["epoch", "train_scale_loss", "val_scale_loss", "val_scale_mse", "val_scale_mae", "val_scale_corr", "val_scale_mean", "val_target_mean", "val_weight_mean"]
    best = float("inf")
    with (args.output/"losses_and_scale_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for epoch in range(1, args.epochs+1):
            net.train(); total = count = 0
            for lr, hr, _ in train:
                lr, hr = lr.to(device), hr.to(device); optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    base, pred, gt = frozen(lr, hr, edsr, residual); target, energy = ridge_target(gt, pred, ridge, lo, hi)
                estimate = pred_scale(net, base.detach(), pred.detach(), lo, hi)
                loss = scale_loss(estimate, target.detach(), energy.detach(), ridge, weighted)
                loss.backward(); optimizer.step(); total += loss.item()*len(lr); count += len(lr)
            stats = validate(net, edsr, residual, val, device, ridge, lo, hi, weighted)
            row = {"epoch": epoch, "train_scale_loss": total/count, **stats}; writer.writerow(row); handle.flush()
            print(f'Ver16-{args.variant} {epoch:03d}/{args.epochs} train={total/count:.5f} val={stats["val_scale_loss"]:.5f} corr={stats["val_scale_corr"]:.3f} MAE={stats["val_scale_mae"]:.3f}', flush=True)
            save = {"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "val": stats}
            torch.save(save, args.output/"last.pt")
            if stats["val_scale_loss"] < best:
                best = stats["val_scale_loss"]; torch.save(save, args.output/"best.pt")

    chosen = torch.load(args.output/"best.pt", map_location=device, weights_only=False); net.load_state_dict(chosen["model"]); net.eval()
    rows, panels, predictions, targets, weights = [], [], [], [], []
    method = f"Ver16-{args.variant} learned ridge scale"
    pred_dir = args.output/"test_predictions"; pred_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for lr, hr, names in test:
            lr, hr = lr.to(device), hr.to(device); base, pred, gt = frozen(lr, hr, edsr, residual)
            target, energy = ridge_target(gt, pred, ridge, lo, hi); estimate = pred_scale(net, base, pred, lo, hi)
            weight = energy/(energy+ridge); predictions.append(estimate.cpu()); targets.append(target.cpu()); weights.append(weight.cpu())
            dense = F.interpolate(estimate, scale_factor=8, mode="nearest")[0,0].cpu().numpy()
            bn, pn, hn = base[0,0].cpu().numpy(), pred[0,0].cpu().numpy(), hr[0,0].cpu().numpy(); name = names[0]
            out = np.round(np.clip(bn+dense*pn, 0, 1)*255).astype(np.uint8).astype(np.float32)/255
            roi = tumor_mask(ROOT/"Paired", name)
            rows.append({"method": method, "file": name, **metric(out, hn, roi), "scale_mean": float(dense.mean()), "scale_roi": float(dense[roi].mean()), "scale_nonroi": float(dense[~roi].mean())})
            Image.fromarray((out*255).astype(np.uint8)).save(pred_dir/f"{name}_pred.png")
            canvas = Image.new("L", (768, 292), 255); draw = ImageDraw.Draw(canvas)
            for i, (im, title) in enumerate(((lr[0,0].cpu().numpy(), "LR"), (out, method), (hn, "GT HR"))):
                canvas.paste(Image.fromarray((np.clip(im,0,1)*255).astype(np.uint8)), (i*256,36)); draw.text((i*256+8,10), title, fill=0)
            draw.text((8,274), name, fill=0); canvas.save(pred_dir/f"{name}_comparison.png"); panels.append(canvas)
    sheet = Image.new("L", (768, 292*len(panels)), 255)
    for i, panel in enumerate(panels): sheet.paste(panel, (0, i*292))
    sheet.save(args.output/"test_comparison_all.png")
    predictions, targets, weights = map(torch.cat, (predictions, targets, weights))
    summary = {"variant": args.variant, "best_epoch": chosen["epoch"], "ridge_lambda": ridge,
               "test_scale_mse": float(F.mse_loss(predictions, targets)), "test_scale_mae": float(F.l1_loss(predictions, targets)),
               "test_scale_corr": corr(predictions, targets), "test_weight_mean": float(weights.mean()),
               "target_clip_zero_fraction": float((targets <= lo+1e-7).float().mean()),
               "target_clip_max_fraction": float((targets >= hi-1e-7).float().mean()), "methods": [summarize(rows, method)]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output/"test_metrics_per_image.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    with (args.output/"test_metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        record = summary["methods"][0]; writer = csv.DictWriter(f, fieldnames=record); writer.writeheader(); writer.writerow(record)
    data = list(csv.DictReader((args.output/"losses_and_scale_diagnostics.csv").open())); x = [int(z["epoch"]) for z in data]
    fig, axes = plt.subplots(1,3,figsize=(15,4)); axes[0].plot(x,[float(z["train_scale_loss"]) for z in data],label="train"); axes[0].plot(x,[float(z["val_scale_loss"]) for z in data],label="val")
    axes[1].plot(x,[float(z["val_scale_corr"]) for z in data]); axes[2].plot(x,[float(z["val_scale_mean"]) for z in data],label="pred"); axes[2].plot(x,[float(z["val_target_mean"]) for z in data],label="target")
    for ax, title in zip(axes,("Scale loss","Validation scale correlation","Scale mean")): ax.set_title(title); ax.grid(alpha=.25); ax.set_xlabel("Epoch"); ax.legend() if len(ax.lines)>1 else None
    fig.tight_layout(); fig.savefig(args.output/"loss_and_scale_diagnostics.png",dpi=180); plt.close(fig)
    (args.output/"train_local_scale_ver16_frozen.py").write_text(Path(__file__).read_text())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
