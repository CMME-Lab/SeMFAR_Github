"""Plot two grayscale images and their pixel-wise differences."""

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def infer_hr_mask_path(hr_image_path: str) -> str:
    """Infer the paired HR label PKL from an HR scan PNG path."""
    scan_name = os.path.basename(hr_image_path)
    if not (scan_name.startswith("scan_") and scan_name.endswith("_HR.png")):
        raise ValueError(f"Cannot infer HR mask from filename: {scan_name}")
    sample = scan_name[len("scan_") : -len("_HR.png")]
    variant_root = os.path.dirname(os.path.dirname(os.path.abspath(hr_image_path)))
    return os.path.join(
        variant_root, "labels", f"label_{sample}_HR.pkl"
    )


def load_tumor_mask(path: str, shape) -> np.ndarray:
    with open(path, "rb") as source:
        data = pickle.load(source)
    if isinstance(data, dict):
        if "clean_phantom_binary" not in data:
            raise KeyError(f"clean_phantom_binary is missing from {path}")
        data = data["clean_phantom_binary"]
    mask = np.asarray(data) > 0.5
    if mask.shape != shape:
        raise ValueError(f"HR tumor mask shape mismatch: {mask.shape} vs {shape}")
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"HR tumor mask has fewer than two pixels: {path}")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("output")
    parser.add_argument("--first-label", default="First")
    parser.add_argument("--second-label", default="Second")
    parser.add_argument(
        "--hr-mask",
        help="HR label PKL; inferred automatically from the first HR PNG",
    )
    args = parser.parse_args()

    first_u8 = load_gray(args.first)
    second_u8 = load_gray(args.second)
    if first_u8.shape != second_u8.shape:
        raise ValueError(
            f"Image shapes must match: {first_u8.shape} vs {second_u8.shape}"
        )

    first = first_u8.astype(np.float32)
    second = second_u8.astype(np.float32)
    signed = first - second
    absolute = np.abs(signed)
    limit = max(float(np.max(absolute)), 1.0)

    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(signed ** 2)))
    max_abs = float(np.max(absolute))
    hr_mask_path = args.hr_mask or infer_hr_mask_path(args.first)
    tumor_mask = load_tumor_mask(hr_mask_path, first.shape)
    tumor_pixels = int(np.count_nonzero(tumor_mask))
    correlation = float(
        np.corrcoef(first[tumor_mask], second[tumor_mask])[0, 1]
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, image, title in (
        (axes[0, 0], first, args.first_label),
        (axes[0, 1], second, args.second_label),
    ):
        shown = ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04, label="8-bit intensity")

    signed_plot = axes[0, 2].imshow(
        signed, cmap="seismic", vmin=-limit, vmax=limit
    )
    axes[0, 2].set_title(f"Signed difference: {args.first_label} - {args.second_label}")
    axes[0, 2].axis("off")
    fig.colorbar(
        signed_plot, ax=axes[0, 2], fraction=0.046, pad=0.04,
        label="intensity difference",
    )

    abs_plot = axes[1, 0].imshow(absolute, cmap="inferno", vmin=0, vmax=limit)
    axes[1, 0].set_title("Absolute difference")
    axes[1, 0].axis("off")
    fig.colorbar(
        abs_plot, ax=axes[1, 0], fraction=0.046, pad=0.04,
        label="absolute intensity difference",
    )

    axes[1, 1].hist(signed.ravel(), bins=101, color="steelblue")
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Signed difference histogram")
    axes[1, 1].set_xlabel("HR - LR intensity")
    axes[1, 1].set_ylabel("pixel count")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.95,
        "Pixel-wise statistics\n\n"
        f"Shape: {first.shape[1]} x {first.shape[0]}\n"
        f"MAE: {mae:.3f} / 255\n"
        f"RMSE: {rmse:.3f} / 255\n"
        f"Max |difference|: {max_abs:.0f} / 255\n"
        f"HR tumor pixels: {tumor_pixels}\n"
        f"Tumor-only Pearson correlation: {correlation:.5f}",
        va="top",
        family="monospace",
        fontsize=13,
    )

    fig.suptitle("Pixel-wise HR/LR comparison", fontsize=16)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"shape={first_u8.shape}")
    print(f"mae={mae:.6f}")
    print(f"rmse={rmse:.6f}")
    print(f"max_abs={max_abs:.6f}")
    print(f"hr_tumor_pixels={tumor_pixels}")
    print(f"tumor_only_correlation={correlation:.8f}")
    print(f"hr_mask={hr_mask_path}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
