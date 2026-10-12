#!/usr/bin/env python3
"""Select and verify the exact source masks used by the data10 dataset.

Raw source images are deliberately not included in this repository.  After
downloading the BUSI archive and the expanded-mask archive yourself, this
script copies only the entries in metadata/data10_mask_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "metadata" / "data10_mask_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--busi-root", type=Path, required=True,
                        help="Extracted Dataset_BUSI_with_GT directory.")
    parser.add_argument("--zenodo-root", type=Path, required=True,
                        help="Expanded-mask directory containing 'benign (N)_mask.png'.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Destination for the selected masks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    rows = list(csv.DictReader(args.manifest.open("r", newline="", encoding="utf-8")))
    if len(rows) != 1038:
        raise RuntimeError(f"Expected 1,038 manifest rows, found {len(rows)}.")

    roots = {"busi": args.busi_root, "zenodo_expanded": args.zenodo_root}
    report = {"manifest_entries": len(rows), "copied": 0, "already_verified": 0, "missing": [], "hash_mismatch": []}

    for row in rows:
        source_root = roots.get(row["source"])
        if source_root is None:
            raise RuntimeError(f"Unknown source group: {row['source']}")
        source = source_root / Path(row["relative_path"])
        destination = args.output_root / Path(row["relative_path"])

        if not source.is_file():
            report["missing"].append(str(source))
            continue
        actual_hash = sha256(source)
        if actual_hash != row["sha256"]:
            report["hash_mismatch"].append({"path": str(source), "expected": row["sha256"], "actual": actual_hash})
            continue
        if destination.is_file() and sha256(destination) == actual_hash:
            report["already_verified"] += 1
            continue
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        report["copied"] += 1

    print(json.dumps(report, indent=2))
    if report["missing"] or report["hash_mismatch"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
