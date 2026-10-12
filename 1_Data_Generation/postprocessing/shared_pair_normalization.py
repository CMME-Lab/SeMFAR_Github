"""Post-process one HR/LR pair using a shared intensity transform.

The original post-processor normalizes each scan line in log_compression and
then min-max normalizes each final image independently.  This script removes
both pair-breaking normalizations: it applies the same absolute log transform
to HR/LR envelopes and then uses one joint min/max range for gamma correction.
Existing simulation and final-output files are not modified.
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import scipy.io

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import core.postprocess_bmode as post_module
from core.postprocess_bmode import (
    PythonSimulationPostProcessor,
)
from config.pipeline_options import PostProcessingConfig


SIM_ROOT = os.path.join(
    PROJECT, "super_resolution", "python_simulation_results_hr_derived_lr"
)
OUT_ROOT = os.path.join(
    PROJECT, "super_resolution", "final_training_data_hr_derived_lr_shared_post_db40_gain"
)
SAMPLE = "benign_5368"
NOISE_LEVEL = "low"
SHARED_GAMMA = 1.0
DB_DYNAMIC_RANGE = 40.0
REFERENCE_PERCENTILE = 99.9


class IdentityGammaPostProcessor(PythonSimulationPostProcessor):
    """Inputs to _save_processed_data are already jointly gamma-corrected."""

    def _gamma_correction(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=np.float32)


def absolute_log_compression(signal, cf, normalize=False):
    """Defer compression so HR/LR can share one amplitude reference."""
    return np.maximum(np.asarray(signal, dtype=np.float64), 0.0)


def load_and_process(variant: str, suffix: str):
    config = PostProcessingConfig(
        target_shape=(256, 256),
        show_preview=False,
        sc_w_x=1.0 if variant == "sc1.0" else 0.5,
        sc_w_y=1.0 if variant == "sc1.0" else 0.5,
        file_suffix=suffix,
    )
    processor = IdentityGammaPostProcessor(config)
    sample_folder = os.path.join(SIM_ROOT, variant, SAMPLE)
    sim_file = os.path.join(sample_folder, f"simulation_{NOISE_LEVEL}.mat")
    phantom_file = os.path.join(sample_folder, f"phantom_{NOISE_LEVEL}.mat")
    sim = scipy.io.loadmat(sim_file)
    phantom = scipy.io.loadmat(phantom_file)

    scan_lines = sim["scan_lines"]
    sound_speed = sim["sound_speed_map"]
    density = sim["density_map"]
    semantic = phantom["semantic_map"]
    dt = float(np.asarray(sim["kgrid_dt"]).squeeze())
    nt = int(np.asarray(sim["kgrid_Nt"]).squeeze())
    element_width = int(np.asarray(sim["element_width"]).squeeze())

    processed = processor._process_scan_lines(
        scan_lines, sound_speed, semantic, dt, nt, element_width
    )
    processed["simulation_metadata"] = processor._extract_simulation_metadata(
        sim, scan_lines, sound_speed, element_width
    )
    processed["simulation_metadata"].update(
        {
            "paired_postprocessing": True,
            "log_compression_normalization": "none",
            "final_normalization": "joint_hr_lr_minmax",
        }
    )
    processed["raw_simulation"] = {
        "scan_lines": scan_lines,
        "kgrid_dt": dt,
        "kgrid_Nt": nt,
        "full_sound_speed_map": sound_speed,
        "full_density_map": density,
        "full_semantic_map": semantic,
    }
    return processor, processed


def jointly_db_compress(hr: np.ndarray, lr: np.ndarray, gamma: float):
    hr_reference = float(np.percentile(hr, REFERENCE_PERCENTILE))
    lr_reference = float(np.percentile(lr, REFERENCE_PERCENTILE))
    hr_reference = max(hr_reference, np.finfo(np.float64).tiny)
    lr_reference = max(lr_reference, np.finfo(np.float64).tiny)
    lr_gain = hr_reference / lr_reference
    reference = hr_reference
    floor_db = -float(DB_DYNAMIC_RANGE)

    def transform(image):
        ratio = np.maximum(image, reference * 1e-12) / reference
        image_db = 20.0 * np.log10(ratio)
        normalized = np.clip((image_db - floor_db) / -floor_db, 0.0, 1.0)
        return np.power(normalized, gamma).astype(np.float32)

    return transform(hr), transform(lr * lr_gain), reference, lr_gain


def save_png(pkl_path: str, png_path: str):
    with open(pkl_path, "rb") as source:
        data = pickle.load(source)
    image = np.asarray(data["noisy_us_scan_b_mode"], dtype=np.float64)
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    if not cv2.imwrite(png_path, image_u8):
        raise RuntimeError(f"Could not save {png_path}")


def main():
    global SAMPLE, SIM_ROOT, OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-name", default=SAMPLE)
    parser.add_argument("--simulation-root", default=SIM_ROOT)
    parser.add_argument("--output-root", default=OUT_ROOT)
    args = parser.parse_args()
    SAMPLE = args.sample_name
    SIM_ROOT = os.path.abspath(args.simulation_root)
    OUT_ROOT = os.path.abspath(args.output_root)

    original_log_compression = post_module.log_compression
    post_module.log_compression = absolute_log_compression
    try:
        hr_processor, hr = load_and_process("sc1.0", "_HR")
        lr_processor, lr = load_and_process("sc0.5", "_LR")
    finally:
        post_module.log_compression = original_log_compression

    gamma = SHARED_GAMMA
    hr["b_mode"], lr["b_mode"], b_reference, b_lr_gain = jointly_db_compress(
        hr["b_mode"], lr["b_mode"], gamma
    )
    hr["harmonic"], lr["harmonic"], h_reference, h_lr_gain = jointly_db_compress(
        hr["harmonic"], lr["harmonic"], gamma
    )

    outputs = {}
    for variant, suffix, processor, processed in (
        ("sc1.0", "_HR", hr_processor, hr),
        ("sc0.5", "_LR", lr_processor, lr),
    ):
        variant_out = os.path.join(OUT_ROOT, variant)
        scans = os.path.join(variant_out, "scans")
        labels = os.path.join(variant_out, "labels")
        os.makedirs(scans, exist_ok=True)
        os.makedirs(labels, exist_ok=True)
        saved = processor._save_processed_data(processed, SAMPLE, scans, labels)
        png = os.path.join(scans, f"scan_{SAMPLE}{suffix}.png")
        save_png(saved["scan_file"], png)
        outputs[variant] = {
            "scan_file": saved["scan_file"],
            "label_file": saved["label_file"],
            "png": png,
        }

    metadata = {
        "sample": SAMPLE,
        "source_simulation_root": SIM_ROOT,
        "output_root": OUT_ROOT,
        "log_transform": "20*log10(envelope/shared_pair_reference)",
        "per_scan_line_normalization": False,
        "joint_final_normalization": "one LR global gain, then shared dB reference/range",
        "gamma": gamma,
        "dynamic_range_db": DB_DYNAMIC_RANGE,
        "reference_percentile": REFERENCE_PERCENTILE,
        "b_mode_shared_reference": b_reference,
        "b_mode_lr_global_gain": b_lr_gain,
        "harmonic_shared_reference": h_reference,
        "harmonic_lr_global_gain": h_lr_gain,
        "outputs": outputs,
    }
    metadata_path = os.path.join(
        OUT_ROOT, f"shared_postprocessing_metadata_{SAMPLE}.json"
    )
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
