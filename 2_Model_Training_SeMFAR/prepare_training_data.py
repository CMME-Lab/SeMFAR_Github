"""Convert Part 1 HR/LR outputs into flat Paired and Unpaired training folders."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


PAIR_PATTERN = re.compile(r"^scan_(.+)_HR\.png$")


def sample_sort_key(sample: str):
    match = re.search(r"(\d+)$", sample)
    return (sample[: match.start()] if match else sample, int(match.group(1)) if match else -1)


def normalize_sample(value: str) -> str:
    value = value.strip()
    if not value or value.startswith("#"):
        return ""
    value = Path(value).stem
    if value.startswith("scan_"):
        value = value[5:]
    value = re.sub(r"_(HR|LR)$", "", value)
    if value.isdigit():
        value = f"benign_{value}"
    return value


def read_list(path: Path | None) -> List[str]:
    if path is None:
        return []
    values = [normalize_sample(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    return [value for value in values if value]


def discover_complete_pairs(generation_root: Path, require_pkl: bool) -> List[str]:
    final_root = generation_root / "final_training_data_hr_derived_lr_shared_post_db40_gain_diff120"
    hr_root = final_root / "sc1.0" / "scans"
    lr_root = final_root / "sc0.5" / "scans"
    samples = []
    for hr_png in hr_root.glob("scan_*_HR.png"):
        match = PAIR_PATTERN.match(hr_png.name)
        if not match:
            continue
        sample = match.group(1)
        files = [lr_root / f"scan_{sample}_LR.png"]
        if require_pkl:
            files.extend(
                [
                    hr_root / f"scan_{sample}_HR.pkl",
                    lr_root / f"scan_{sample}_LR.pkl",
                ]
            )
        if all(path.is_file() and path.stat().st_size > 0 for path in files):
            samples.append(sample)
    return sorted(set(samples), key=sample_sort_key)


def load_final_reprocessor(part1_root: Path):
    script = part1_root / "postprocessing" / "final_db40_pair_reprocess.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    sys.path.insert(0, str(part1_root))
    spec = importlib.util.spec_from_file_location("semfar_final_db40", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def choose_samples(
    available: List[str],
    paired_count: int,
    unpaired_count: int,
    seed: int,
    paired_list: List[str],
    unpaired_list: List[str],
):
    available_set = set(available)
    missing = sorted((set(paired_list) | set(unpaired_list)) - available_set, key=sample_sort_key)
    if missing:
        raise ValueError(f"Requested samples are unavailable: {missing[:10]}")
    if set(paired_list) & set(unpaired_list):
        raise ValueError("Paired and Unpaired lists overlap")
    rng = random.Random(seed)
    remaining = [sample for sample in available if sample not in paired_list and sample not in unpaired_list]
    rng.shuffle(remaining)
    paired = list(paired_list)
    if len(paired) > paired_count:
        raise ValueError("Paired list is longer than --paired-count")
    paired.extend(remaining[: paired_count - len(paired)])
    remaining = [sample for sample in remaining if sample not in set(paired)]
    unpaired = list(unpaired_list)
    requested_unpaired = len(remaining) if unpaired_count == 0 else unpaired_count
    if len(unpaired) > requested_unpaired:
        raise ValueError("Unpaired list is longer than --unpaired-count")
    unpaired.extend(remaining[: requested_unpaired - len(unpaired)])
    if len(paired) != paired_count:
        raise ValueError(f"Need {paired_count} pairs but found {len(paired)}")
    if not unpaired:
        raise ValueError("At least one Unpaired LR image is required for EMA")
    return paired, unpaired


def ensure_empty_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.rglob("*.png")):
        raise FileExistsError(
            f"Output already contains PNG files: {output_root}. Choose a new directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)


def copy_existing(
    final_root: Path, paired: Iterable[str], unpaired: Iterable[str], output_root: Path
) -> None:
    paired_root, unpaired_root = output_root / "Paired", output_root / "Unpaired"
    paired_root.mkdir(parents=True, exist_ok=True)
    unpaired_root.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(paired, 1):
        shutil.copy2(
            final_root / "sc1.0" / "scans" / f"scan_{sample}_HR.png",
            paired_root / f"scan_{sample}_HR.png",
        )
        shutil.copy2(
            final_root / "sc0.5" / "scans" / f"scan_{sample}_LR.png",
            paired_root / f"scan_{sample}_LR.png",
        )
        print(f"PAIRED {index}: {sample}", flush=True)
    for index, sample in enumerate(unpaired, 1):
        shutil.copy2(
            final_root / "sc0.5" / "scans" / f"scan_{sample}_LR.png",
            unpaired_root / f"scan_{sample}_LR.png",
        )
        print(f"UNPAIRED {index}: {sample}", flush=True)


def reprocess_final_db40(
    module,
    final_root: Path,
    paired: Iterable[str],
    unpaired: Iterable[str],
    output_root: Path,
) -> None:
    paired_root, unpaired_root = output_root / "Paired", output_root / "Unpaired"
    temp_root = output_root / "_final_db40_temp"
    paired_root.mkdir(parents=True, exist_ok=True)
    unpaired_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(paired, 1):
        temp = temp_root / sample
        temp.mkdir(parents=True, exist_ok=True)
        _, hr_png = module.process_one(
            final_root / "sc1.0" / "scans" / f"scan_{sample}_HR.pkl",
            sample,
            "HR",
            temp,
        )
        _, lr_png = module.process_one(
            final_root / "sc0.5" / "scans" / f"scan_{sample}_LR.pkl",
            sample,
            "LR",
            temp,
        )
        shutil.copy2(hr_png, paired_root / hr_png.name)
        shutil.copy2(lr_png, paired_root / lr_png.name)
        shutil.rmtree(temp)
        print(f"FINAL-DB40 PAIRED {index}: {sample}", flush=True)
    for index, sample in enumerate(unpaired, 1):
        temp = temp_root / sample
        temp.mkdir(parents=True, exist_ok=True)
        _, lr_png = module.process_one(
            final_root / "sc0.5" / "scans" / f"scan_{sample}_LR.pkl",
            sample,
            "LR",
            temp,
        )
        shutil.copy2(lr_png, unpaired_root / lr_png.name)
        shutil.rmtree(temp)
        print(f"FINAL-DB40 UNPAIRED {index}: {sample}", flush=True)
    if temp_root.exists() and not any(temp_root.iterdir()):
        temp_root.rmdir()


def create_split(paired: List[str], seed: int, output_root: Path) -> dict:
    if len(paired) != 50:
        raise ValueError("Runnable comparison currently requires exactly 50 paired cases")
    names = [f"scan_{sample}" for sample in paired]
    rng = random.Random(seed)
    rng.shuffle(names)
    split = {
        "selection_seed": seed,
        "train": names[:30],
        "validation": names[30:40],
        "test": names[40:50],
    }
    (output_root / "split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8", newline="\n"
    )
    return split


def validate_output(output_root: Path, paired: List[str], unpaired: List[str]) -> None:
    paired_root, unpaired_root = output_root / "Paired", output_root / "Unpaired"
    missing = []
    for sample in paired:
        for tag in ("HR", "LR"):
            path = paired_root / f"scan_{sample}_{tag}.png"
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    for sample in unpaired:
        path = unpaired_root / f"scan_{sample}_LR.png"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        raise RuntimeError(f"Prepared dataset is incomplete; first missing files: {missing[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument(
        "--part1-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "1_Data_Generation",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--paired-count", type=int, default=50)
    parser.add_argument(
        "--unpaired-count", type=int, default=0, help="0 uses every remaining sample"
    )
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--paired-list", type=Path)
    parser.add_argument("--unpaired-list", type=Path)
    parser.add_argument(
        "--postprocess-mode",
        choices=("final-db40", "existing"),
        default="final-db40",
        help="final-db40 matches data10; existing is faster but keeps Part 1 shared-pair display",
    )
    args = parser.parse_args()

    args.generation_root = args.generation_root.resolve()
    args.part1_root = args.part1_root.resolve()
    args.output_root = args.output_root.resolve()
    require_pkl = args.postprocess_mode == "final-db40"
    available = discover_complete_pairs(args.generation_root, require_pkl)
    paired, unpaired = choose_samples(
        available,
        args.paired_count,
        args.unpaired_count,
        args.selection_seed,
        read_list(args.paired_list),
        read_list(args.unpaired_list),
    )
    ensure_empty_output(args.output_root)
    final_root = (
        args.generation_root
        / "final_training_data_hr_derived_lr_shared_post_db40_gain_diff120"
    )
    if args.postprocess_mode == "final-db40":
        reprocessor = load_final_reprocessor(args.part1_root)
        reprocess_final_db40(reprocessor, final_root, paired, unpaired, args.output_root)
    else:
        copy_existing(final_root, paired, unpaired, args.output_root)
    split = create_split(paired, args.selection_seed, args.output_root)
    validate_output(args.output_root, paired, unpaired)
    manifest = {
        "generation_root": str(args.generation_root),
        "part1_root": str(args.part1_root),
        "postprocess_mode": args.postprocess_mode,
        "selection_seed": args.selection_seed,
        "paired_count": len(paired),
        "unpaired_count": len(unpaired),
        "paired_samples": paired,
        "unpaired_samples": unpaired,
        "split": split,
    }
    (args.output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: manifest[key] for key in ("paired_count", "unpaired_count", "postprocess_mode")}, indent=2))


if __name__ == "__main__":
    main()

