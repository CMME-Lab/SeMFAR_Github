"""Train the frozen residual predictor used by SeMFAR on the current Paired data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    EDSR,
    PairDataset,
    ResidualPredictor,
    charbonnier,
    checkpoint_state,
    discover_pairs,
    gaussian,
    load_split,
    seed_everything,
    split_records,
)


@torch.no_grad()
def validate(model, upstream, loader, device, sigma, lambda_full):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        coarse = upstream(lr)
        target = hr - coarse
        prediction = model(lr, coarse)
        low = charbonnier(gaussian(prediction, sigma), gaussian(target, sigma))
        full = charbonnier(prediction, target)
        loss = low + lambda_full * full
        totals += np.array([loss.item(), low.item(), full.item()]) * len(lr)
        count += len(lr)
    return totals / count


@torch.no_grad()
def calibrate_ridge(model, upstream, loader, device, sigma):
    """Derive the SeMFAR ridge constant from this run's validation pairs."""
    model.eval()
    energies = []
    numerators = []
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        coarse = upstream(lr)
        prediction = gaussian(model(lr, coarse), sigma)
        target = gaussian(hr - coarse, sigma)
        energies.append(torch.nn.functional.avg_pool2d(prediction.square(), 8, 8).flatten().cpu())
        numerators.append(torch.nn.functional.avg_pool2d(target * prediction, 8, 8).flatten().cpu())
    energy = torch.cat(energies).numpy().astype(np.float64)
    numerator = torch.cat(numerators).numpy().astype(np.float64)
    positive = energy[energy > 0]
    if positive.size == 0:
        raise RuntimeError("Residual predictor produced no positive validation energy")
    ridge = float(np.quantile(positive, 0.10))
    raw_target = numerator / (energy + ridge)
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.99]
    return {
        "variant": "new-data validation calibration",
        "ridge_rule": "validation positive residual-energy q10",
        "ridge_lambda": ridge,
        "energy_quantiles": {str(q): float(np.quantile(positive, q)) for q in quantiles},
        "target_quantiles": {str(q): float(np.quantile(raw_target, q)) for q in quantiles},
        "scale_bounds": [0.0, 3.0],
        "source": "current seed validation split and newly trained upstream/residual checkpoints",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--upstream-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--lambda-full", type=float, default=0.1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = discover_pairs(args.paired)
    split = load_split(args.split, records)
    sets = split_records(split, records)
    (args.output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    checkpoint = torch.load(args.upstream_checkpoint, map_location=device, weights_only=False)
    upstream = EDSR(checkpoint.get("args", {}).get("residual_scale", 0.1)).to(device)
    upstream.load_state_dict(checkpoint_state(checkpoint))
    upstream.eval().requires_grad_(False)
    model = ResidualPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = DataLoader(PairDataset(sets["train"], True), args.batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(sets["validation"]), args.batch_size)
    test_loader = DataLoader(PairDataset(sets["test"]), 1)
    fields = ["epoch", "train_total", "train_low", "train_full", "val_total", "val_low", "val_full"]
    best, early_best, early_epoch = float("inf"), float("inf"), 0
    with (args.output / "losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals = np.zeros(3, dtype=np.float64)
            count = 0
            for lr, hr, _ in train_loader:
                lr, hr = lr.to(device), hr.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    coarse = upstream(lr)
                    target = hr - coarse
                    target_low = gaussian(target, args.sigma)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    prediction = model(lr, coarse)
                    low = charbonnier(gaussian(prediction, args.sigma), target_low)
                    full = charbonnier(prediction, target)
                    loss = low + args.lambda_full * full
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                totals += np.array([loss.item(), low.item(), full.item()]) * len(lr)
                count += len(lr)
            train_values = totals / count
            val_values = validate(model, upstream, val_loader, device, args.sigma, args.lambda_full)
            row = dict(zip(fields, [epoch, *train_values, *val_values]))
            writer.writerow(row)
            handle.flush()
            print(f"Residual {epoch:03d}/{args.epochs} train={train_values[0]:.6f} val={val_values[0]:.6f}", flush=True)
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "upstream_checkpoint": str(args.upstream_checkpoint),
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
    model.load_state_dict(chosen["model"])
    test_values = validate(model, upstream, test_loader, device, args.sigma, args.lambda_full)
    (args.output / "test_metrics.json").write_text(
        json.dumps(
            {"best_epoch": chosen["epoch"], "total": float(test_values[0]), "low": float(test_values[1]), "full": float(test_values[2])},
            indent=2,
        ),
        encoding="utf-8",
    )
    ridge_config = calibrate_ridge(model, upstream, val_loader, device, args.sigma)
    (args.output / "ridge_and_target_config.json").write_text(
        json.dumps(ridge_config, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
