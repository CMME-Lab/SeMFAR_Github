"""Apply one linear scale to an HR/LR pair to target max pixel difference."""

import argparse
import os
import pickle
import shutil

import cv2
import numpy as np


PROJECT = r"C:\Users\CMME260629\projects_cmme\SuperResolutionAI"
SOURCE = os.path.join(
    PROJECT,
    "super_resolution",
    "final_training_data_hr_derived_lr_shared_post_db40_gain",
)
OUTPUT = os.path.join(
    PROJECT,
    "super_resolution",
    "final_training_data_hr_derived_lr_shared_post_db40_gain_diff120",
)
SAMPLE = "benign_5368"
TARGET_MAX_DIFFERENCE = 120.0


def load_png(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-name", default=SAMPLE)
    parser.add_argument("--source-root", default=SOURCE)
    parser.add_argument("--output-root", default=OUTPUT)
    parser.add_argument(
        "--target-max-difference", type=float, default=TARGET_MAX_DIFFERENCE
    )
    args = parser.parse_args()
    sample = args.sample_name
    source = os.path.abspath(args.source_root)
    output = os.path.abspath(args.output_root)

    hr_png_source = os.path.join(
        source, "sc1.0", "scans", f"scan_{sample}_HR.png"
    )
    lr_png_source = os.path.join(
        source, "sc0.5", "scans", f"scan_{sample}_LR.png"
    )
    current_max = float(
        np.max(np.abs(load_png(hr_png_source) - load_png(lr_png_source)))
    )
    scale = min(1.0, args.target_max_difference / max(current_max, 1.0))

    for variant, suffix in (("sc1.0", "_HR"), ("sc0.5", "_LR")):
        source_scan = os.path.join(
            source, variant, "scans", f"scan_{sample}{suffix}.pkl"
        )
        output_scans = os.path.join(output, variant, "scans")
        output_labels = os.path.join(output, variant, "labels")
        os.makedirs(output_scans, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)

        with open(source_scan, "rb") as source_file:
            scan_data = pickle.load(source_file)
        scan_data["noisy_us_scan_b_mode"] = (
            np.asarray(scan_data["noisy_us_scan_b_mode"], dtype=np.float32) * scale
        )
        scan_data["noisy_us_scan_harmonic"] = (
            np.asarray(scan_data["noisy_us_scan_harmonic"], dtype=np.float32) * scale
        )
        metadata = scan_data.setdefault("simulation_metadata", {})
        metadata["shared_linear_brightness_scale"] = float(scale)
        metadata["target_max_pixel_difference"] = float(
            args.target_max_difference
        )

        output_scan = os.path.join(
            output_scans, f"scan_{sample}{suffix}.pkl"
        )
        with open(output_scan, "wb") as output_file:
            pickle.dump(scan_data, output_file)

        output_png = os.path.join(
            output_scans, f"scan_{sample}{suffix}.png"
        )
        image = np.clip(
            np.rint(scan_data["noisy_us_scan_b_mode"] * 255.0), 0, 255
        ).astype(np.uint8)
        if not cv2.imwrite(output_png, image):
            raise RuntimeError(f"Could not write {output_png}")

        source_label = os.path.join(
            source, variant, "labels", f"label_{sample}{suffix}.pkl"
        )
        if os.path.exists(source_label):
            shutil.copy2(
                source_label,
                os.path.join(output_labels, os.path.basename(source_label)),
            )

    print(f"sample={sample}")
    print(f"source_max_abs_difference={current_max:.6f}")
    print(f"linear_scale={scale:.9f}")
    print(f"target_max_abs_difference={args.target_max_difference:.6f}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
