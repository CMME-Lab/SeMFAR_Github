"""Train SeMFAR with newly trained paired-data upstream and residual modules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common import (
    EDSR,
    PairDataset,
    ResidualPredictor,
    aggregate_metrics,
    checkpoint_state,
    discover_pairs,
    gaussian,
    gradient_charbonnier,
    image_metrics,
    load_split,
    seed_everything,
    split_records,
)


class AffineHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(65, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, 1),
        )

    def forward(self, echo_features, residual):
        echo = F.adaptive_avg_pool2d(self.project(echo_features[1]), (32, 32))
        pooled = F.avg_pool2d(residual, 8, 8)
        return self.fuse(torch.cat((echo, pooled), dim=1))


def make_echocare(checkpoint: Path, device):
    try:
        from monai.networks.nets.swin_unetr import SwinTransformer
    except ImportError as error:
        raise RuntimeError("MONAI is required for SeMFAR: pip install monai") from error
    encoder = SwinTransformer(
        in_chans=3,
        embed_dim=128,
        window_size=[8, 8],
        patch_size=[2, 2],
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        qkv_bias=True,
        use_checkpoint=False,
        spatial_dims=2,
        use_v2=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state.pop("mask_token", None)
    encoder.load_state_dict(state, strict=True)
    return encoder.to(device).eval().requires_grad_(False)


def ridge_target(target, prediction, ridge, lower, upper, patch=8):
    numerator = F.avg_pool2d(target * prediction, patch, patch)
    energy = F.avg_pool2d(prediction.square(), patch, patch)
    return (numerator / (energy + ridge)).clamp(lower, upper)


def forward_head(head, features, residual, target, ridge, lower, upper):
    raw = head(features, residual)
    scale = lower + (upper - lower) * torch.sigmoid(raw[:, :1])
    bias = 0.1 * torch.tanh(raw[:, 1:2])
    target_scale = ridge_target(target, residual, ridge, lower, upper)
    dense_scale = F.interpolate(scale, size=residual.shape[-2:], mode="nearest")
    dense_bias = F.interpolate(bias, size=residual.shape[-2:], mode="nearest")
    corrected = dense_scale * residual + dense_bias
    scale_loss = F.smooth_l1_loss(scale, target_scale)
    return corrected, scale_loss, scale, bias


@torch.no_grad()
def validate(head, upstream, residual_model, echo, loader, device, ridge, lower, upper):
    head.eval()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        base = upstream(lr)
        residual = gaussian(residual_model(lr, base), 1.5)
        target = gaussian(hr - base, 1.5)
        features = echo(lr.repeat(1, 3, 1, 1))
        corrected, scale_loss, _, _ = forward_head(
            head, features, residual, target, ridge, lower, upper
        )
        final = (base + corrected).clamp(0, 1)
        gradient = 0.2 * gradient_charbonnier(final, hr)
        loss = scale_loss + gradient
        totals += np.array([loss.item(), scale_loss.item(), gradient.item()]) * len(lr)
        count += len(lr)
    return totals / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--upstream-checkpoint", required=True, type=Path)
    parser.add_argument("--residual-checkpoint", required=True, type=Path)
    parser.add_argument("--echocare-checkpoint", required=True, type=Path)
    parser.add_argument("--ridge-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.effective_batch_size % args.batch_size:
        raise ValueError("effective batch size must be divisible by batch size")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = discover_pairs(args.paired)
    split = load_split(args.split, records)
    sets = split_records(split, records)
    (args.output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    ridge_config = json.loads(args.ridge_config.read_text(encoding="utf-8"))
    ridge = float(ridge_config["ridge_lambda"])
    lower, upper = map(float, ridge_config["scale_bounds"])

    upstream_checkpoint = torch.load(args.upstream_checkpoint, map_location=device, weights_only=False)
    upstream = EDSR(upstream_checkpoint.get("args", {}).get("residual_scale", 0.1)).to(device)
    upstream.load_state_dict(checkpoint_state(upstream_checkpoint))
    upstream.eval().requires_grad_(False)
    residual_checkpoint = torch.load(args.residual_checkpoint, map_location=device, weights_only=False)
    residual_model = ResidualPredictor().to(device)
    residual_model.load_state_dict(checkpoint_state(residual_checkpoint))
    residual_model.eval().requires_grad_(False)
    echo = make_echocare(args.echocare_checkpoint, device)
    head = AffineHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    train_loader = DataLoader(PairDataset(sets["train"], True), args.batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(sets["validation"]), args.batch_size)
    test_loader = DataLoader(PairDataset(sets["test"]), 1)
    accumulate = args.effective_batch_size // args.batch_size
    fields = ["epoch", "train_total", "train_scale", "train_gradient", "val_total", "val_scale", "val_gradient"]
    best, early_best, early_epoch = float("inf"), float("inf"), 0
    with (args.output / "losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            totals = np.zeros(3, dtype=np.float64)
            count = 0
            for step, (lr, hr, _) in enumerate(train_loader, 1):
                lr, hr = lr.to(device), hr.to(device)
                with torch.no_grad():
                    base = upstream(lr)
                    residual = gaussian(residual_model(lr, base), 1.5)
                    target = gaussian(hr - base, 1.5)
                    features = echo(lr.repeat(1, 3, 1, 1))
                corrected, scale_loss, _, _ = forward_head(
                    head, features, residual, target, ridge, lower, upper
                )
                final = (base.detach() + corrected).clamp(0, 1)
                gradient = 0.2 * gradient_charbonnier(final, hr)
                loss = scale_loss + gradient
                (loss / accumulate).backward()
                if step % accumulate == 0 or step == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                totals += np.array([loss.item(), scale_loss.item(), gradient.item()]) * len(lr)
                count += len(lr)
            train_values = totals / count
            val_values = validate(
                head, upstream, residual_model, echo, val_loader, device, ridge, lower, upper
            )
            row = dict(zip(fields, [epoch, *train_values, *val_values]))
            writer.writerow(row)
            handle.flush()
            print(f"SeMFAR {epoch:03d}/{args.epochs} train={train_values[0]:.6f} val={val_values[0]:.6f}", flush=True)
            state = {
                "epoch": epoch,
                "model": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "ridge_config": ridge_config,
                "val_total": float(val_values[0]),
            }
            torch.save(state, args.output / "last.pt")
            if val_values[0] < best:
                best = float(val_values[0])
                torch.save(state, args.output / "best.pt")
            if val_values[0] < early_best - 1e-4:
                early_best, early_epoch = float(val_values[0]), epoch
            if epoch - early_epoch >= 50:
                break

    chosen = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    head.load_state_dict(chosen["model"])
    head.eval()
    prediction_root = args.output / "test_predictions"
    prediction_root.mkdir(exist_ok=True)
    rows = []
    with torch.no_grad():
        for lr, hr, names in test_loader:
            lr, hr = lr.to(device), hr.to(device)
            base = upstream(lr)
            residual = gaussian(residual_model(lr, base), 1.5)
            target = gaussian(hr - base, 1.5)
            features = echo(lr.repeat(1, 3, 1, 1))
            corrected, _, _, _ = forward_head(
                head, features, residual, target, ridge, lower, upper
            )
            prediction = (base + corrected).clamp(0, 1)[0, 0].cpu().numpy()
            target_np = hr[0, 0].cpu().numpy()
            name = names[0]
            Image.fromarray((prediction * 255).round().astype(np.uint8)).save(
                prediction_root / f"{name}_pred.png"
            )
            rows.append({"file": name, **image_metrics(prediction, target_np)})
    summary = {
        "model": "SeMFAR affine residual modulation",
        "best_epoch": chosen["epoch"],
        **aggregate_metrics(rows),
    }
    (args.output / "test_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output / "test_metrics_per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

