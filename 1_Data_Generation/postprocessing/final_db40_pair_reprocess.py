"""Apply the exact AI3 LR-only DB40/fixed-gain display transform to an HR/LR pair."""

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.postprocess_bmode as post_module
from core.postprocess_bmode import PythonSimulationPostProcessor
from config.pipeline_options import PostProcessingConfig

DB_DYNAMIC_RANGE = 40.0
REFERENCE_PERCENTILE = 99.9
FIXED_DIFF120_GAIN = 0.567379057497988
EXACT_PLOTTER = PROJECT_ROOT / "visualization" / "plot_pair_difference_tumor_roi.py"


class IdentityGammaPostProcessor(PythonSimulationPostProcessor):
    def _gamma_correction(self, image):
        return np.asarray(image, dtype=np.float32)


def absolute_log_compression(signal, cf, normalize=False):
    return np.maximum(np.asarray(signal, dtype=np.float64), 0.0)


def db40(image):
    reference = max(
        float(np.percentile(image, REFERENCE_PERCENTILE)),
        np.finfo(np.float64).tiny,
    )
    ratio = np.maximum(image, reference * 1e-12) / reference
    image_db = 20.0 * np.log10(ratio)
    normalized = np.clip((image_db + DB_DYNAMIC_RANGE) / DB_DYNAMIC_RANGE, 0.0, 1.0)
    return normalized.astype(np.float32), reference


def save_png(pkl_path: Path, png_path: Path):
    with pkl_path.open("rb") as source:
        data = pickle.load(source)
    image = np.asarray(data["noisy_us_scan_b_mode"], dtype=np.float64)
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(png_path), image_u8):
        raise RuntimeError(f"Could not save {png_path}")


def process_one(source_pkl: Path, sample: str, tag: str, output: Path):
    scale = 1.0 if tag == "HR" else 0.5
    with source_pkl.open("rb") as handle:
        data = pickle.load(handle)
    scan_lines = np.asarray(data["scan_lines"])
    sound_speed = np.asarray(data["full_sound_speed_map"])
    density = np.asarray(data["full_density_map"])
    semantic = np.asarray(data["full_semantic_map"])
    dt = float(np.asarray(data["kgrid_dt"]).squeeze())
    nt = int(np.asarray(data["kgrid_Nt"]).squeeze())
    metadata_in = data.get("simulation_metadata", {})
    element_width = int(metadata_in["scan_element_width"])

    processor = IdentityGammaPostProcessor(PostProcessingConfig(
        target_shape=(256, 256), show_preview=False,
        sc_w_x=scale, sc_w_y=scale, file_suffix=f"_{tag}",
    ))
    original = post_module.log_compression
    post_module.log_compression = absolute_log_compression
    try:
        processed = processor._process_scan_lines(
            scan_lines, sound_speed, semantic, dt, nt, element_width
        )
    finally:
        post_module.log_compression = original

    processed["b_mode"], b_ref = db40(processed["b_mode"])
    processed["harmonic"], h_ref = db40(processed["harmonic"])
    processed["b_mode"] *= FIXED_DIFF120_GAIN
    processed["harmonic"] *= FIXED_DIFF120_GAIN
    metadata = dict(metadata_in)
    metadata.update({
        "paired_postprocessing": False,
        "lr_only_style_db40_applied_to_pair": True,
        "lr_only_db40": True,
        "log_compression_normalization": f"{tag.lower()}_99.9_percentile",
        "dynamic_range_db": DB_DYNAMIC_RANGE,
        "fixed_diff120_brightness_gain": FIXED_DIFF120_GAIN,
        "fixed_gain_source": "median_of_50_existing_AI3_diff120_pairs",
        "target_max_pixel_difference_is_estimated": True,
        "b_mode_reference": b_ref,
        "harmonic_reference": h_ref,
    })
    processed["simulation_metadata"] = metadata
    processed["raw_simulation"] = {
        "scan_lines": scan_lines,
        "kgrid_dt": dt,
        "kgrid_Nt": nt,
        "full_sound_speed_map": sound_speed,
        "full_density_map": density,
        "full_semantic_map": semantic,
    }
    scans, labels = output / f"{tag}_work" / "scans", output / f"{tag}_work" / "labels"
    scans.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    saved = processor._save_processed_data(processed, sample, str(scans), str(labels))
    source_saved = Path(saved["scan_file"])
    final_pkl = output / f"scan_{sample}_{tag}.pkl"
    source_saved.replace(final_pkl)
    final_png = output / f"scan_{sample}_{tag}.png"
    save_png(final_pkl, final_png)
    return final_pkl, final_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for tag in ("HR", "LR"):
        outputs[tag] = process_one(
            args.input_dir / f"scan_{args.sample}_{tag}.pkl",
            args.sample, tag, args.output_dir,
        )
    difference = args.output_dir / f"scan_{args.sample}_HR_minus_LR_pixelwise.png"
    subprocess.run([
        sys.executable, str(EXACT_PLOTTER), str(outputs["HR"][1]), str(outputs["LR"][1]),
        str(difference), "--first-label", "HR", "--second-label", "LR",
    ], check=True)
    print(f"COMPLETE {args.sample}", flush=True)


if __name__ == "__main__":
    main()
