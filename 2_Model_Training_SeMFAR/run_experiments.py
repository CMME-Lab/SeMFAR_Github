"""Run the reproducible EMA and SeMFAR experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require_working_cuda() -> None:
    """Reject builds that see the GPU but cannot execute kernels on it."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build, or use "
            "--allow-cpu only if a very slow CPU run is intentional."
        )
    try:
        probe = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        if float((probe + 1).cpu().item()) != 2.0:
            raise RuntimeError("unexpected CUDA probe result")
    except Exception as error:
        raise RuntimeError(
            "PyTorch detects the GPU but cannot execute a CUDA kernel. Install a "
            "PyTorch/CUDA build that supports this GPU's compute capability."
        ) from error


def run(command, dry_run: bool) -> None:
    print("\nRUN:", subprocess.list2cmdline([str(value) for value in command]), flush=True)
    if not dry_run:
        subprocess.run([str(value) for value in command], check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ema", "semfar"),
        default=["ema", "semfar"],
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a real run without CUDA (normally rejected because this workflow is GPU training)",
    )
    args = parser.parse_args()
    paired = args.data_root.resolve() / "Paired"
    unpaired = args.data_root.resolve() / "Unpaired"
    split = args.data_root.resolve() / "split.json"
    if not paired.is_dir() or not split.is_file():
        raise FileNotFoundError("Prepared Paired directory or split.json is missing")
    if "ema" in args.models and not unpaired.is_dir():
        raise FileNotFoundError("Prepared Unpaired directory is missing")
    if not args.dry_run and not args.allow_cpu:
        require_working_cuda()
    args.output_root.mkdir(parents=True, exist_ok=True)

    upstream_script = ROOT / "runnable" / "train_upstream_edsr.py"
    ema_script = ROOT / "02_EMA" / "seed123" / "train_ema_rrad02_adamw_lr1e3.py"
    residual_script = ROOT / "runnable" / "train_residual_predictor.py"
    semfar_script = ROOT / "runnable" / "train_semfar.py"
    echo_checkpoint = ROOT / "04_SeMFAR" / "frozen_dependencies" / "echocare_encoder.pth"

    plan = []
    for seed in args.seeds:
        seed_root = args.output_root / f"seed{seed}"
        upstream_output = seed_root / "SeMFAR_Upstream_EDSR"
        ema_output = seed_root / "EMA"
        residual_output = seed_root / "SeMFAR_Residual_Predictor"
        semfar_output = seed_root / "SeMFAR"
        if "semfar" in args.models:
            command = [
                sys.executable, upstream_script, "--paired", paired, "--split", split,
                "--output", upstream_output, "--epochs", args.epochs,
                "--batch-size", args.batch_size, "--seed", seed,
            ]
            plan.append(command)
            run(command, args.dry_run)
        if "ema" in args.models:
            command = [
                sys.executable, ema_script, "--paired", paired, "--unpaired", unpaired,
                "--split", split, "--output", ema_output, "--epochs", args.epochs,
                "--batch-size", args.batch_size, "--seed", seed,
            ]
            plan.append(command)
            run(command, args.dry_run)
        if "semfar" in args.models:
            command = [
                sys.executable, residual_script, "--paired", paired, "--split", split,
                "--upstream-checkpoint", upstream_output / "best.pt", "--output", residual_output,
                "--epochs", args.epochs, "--batch-size", args.batch_size, "--seed", seed,
            ]
            plan.append(command)
            run(command, args.dry_run)
            command = [
                sys.executable, semfar_script, "--paired", paired, "--split", split,
                "--upstream-checkpoint", upstream_output / "best.pt",
                "--residual-checkpoint", residual_output / "best.pt",
                "--echocare-checkpoint", echo_checkpoint,
                "--ridge-config", residual_output / "ridge_and_target_config.json",
                "--output", semfar_output, "--epochs", args.epochs,
                "--batch-size", min(args.batch_size, 4), "--seed", seed,
            ]
            plan.append(command)
            run(command, args.dry_run)
    (args.output_root / "run_plan.json").write_text(
        json.dumps([[str(value) for value in command] for command in plan], indent=2),
        encoding="utf-8",
    )
    print("\nPFT is not run: its exact original implementation/checkpoint was not present in the archive.")


if __name__ == "__main__":
    main()
