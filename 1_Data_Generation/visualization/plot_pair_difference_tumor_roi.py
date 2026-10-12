"""Plot HR/LR differences using a tumor ROI segmented from the HR PNG itself."""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def segment_tumor_from_hr(hr_u8: np.ndarray) -> np.ndarray:
    """Extract the dominant dark, non-border lesion directly from an HR PNG."""
    height, width = hr_u8.shape
    blurred = cv2.GaussianBlur(hr_u8, (0, 0), 3.0)
    _, dark = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    margin_x = max(5, int(round(width * 0.035)))
    margin_y = max(5, int(round(height * 0.035)))
    dark[:margin_y, :] = 0
    dark[-margin_y:, :] = 0
    dark[:, :margin_x] = 0
    dark[:, -margin_x:] = 0
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    image_area = float(height * width)
    best_index = None
    best_score = -np.inf
    for index in range(1, count):
        x, y, box_w, box_h, area = stats[index]
        area_fraction = float(area) / image_area
        if not 0.008 <= area_fraction <= 0.65:
            continue
        cx, cy = centroids[index]
        normalized_distance = np.hypot(
            (cx - width / 2.0) / (width / 2.0),
            (cy - height / 2.0) / (height / 2.0),
        )
        component = labels == index
        darkness = (255.0 - float(np.mean(hr_u8[component]))) / 255.0
        # Favor a sizable, dark component near the image center. Bounding-box
        # dimensions prevent thin scan-line remnants from winning.
        thickness = min(box_w / width, box_h / height)
        score = (
            np.log1p(float(area))
            * max(darkness, 1e-3)
            * max(thickness, 1e-3)
            * np.exp(-1.5 * normalized_distance)
        )
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        raise ValueError("No plausible tumor component found in HR PNG")
    mask = (labels == best_index).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    if np.count_nonzero(mask) < 2:
        raise ValueError("HR-PNG-derived tumor mask has fewer than two pixels")
    return mask


def safe_correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    first_values = first[mask].astype(np.float64)
    second_values = second[mask].astype(np.float64)
    if first_values.size < 2:
        return float("nan")
    if np.std(first_values) == 0.0 or np.std(second_values) == 0.0:
        return float("nan")
    return float(np.corrcoef(first_values, second_values)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("output")
    parser.add_argument("--first-label", default="HR")
    parser.add_argument("--second-label", default="LR")
    args = parser.parse_args()

    first_u8 = load_gray(args.first)
    second_u8 = load_gray(args.second)
    if first_u8.shape != second_u8.shape:
        raise ValueError(
            f"Image shapes must match: {first_u8.shape} vs {second_u8.shape}"
        )
    tumor_mask = segment_tumor_from_hr(first_u8)
    first = first_u8.astype(np.float32)
    second = second_u8.astype(np.float32)
    signed = first - second
    absolute = np.abs(signed)
    limit = max(float(np.max(absolute)), 1.0)
    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(signed ** 2)))
    max_abs = float(np.max(absolute))
    tumor_pixels = int(np.count_nonzero(tumor_mask))
    correlation = safe_correlation(first, second, tumor_mask)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, image, title in (
        (axes[0, 0], first, args.first_label),
        (axes[0, 1], second, args.second_label),
    ):
        shown = ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04, label="8-bit intensity")
    axes[0, 0].contour(tumor_mask, levels=[0.5], colors=["red"], linewidths=0.8)
    axes[0, 0].set_title(f"{args.first_label} (HR-PNG tumor outline)")

    signed_plot = axes[0, 2].imshow(signed, cmap="seismic", vmin=-limit, vmax=limit)
    axes[0, 2].set_title(f"Signed difference: {args.first_label} - {args.second_label}")
    axes[0, 2].axis("off")
    fig.colorbar(signed_plot, ax=axes[0, 2], fraction=0.046, pad=0.04, label="intensity difference")

    abs_plot = axes[1, 0].imshow(absolute, cmap="inferno", vmin=0, vmax=limit)
    axes[1, 0].set_title("Absolute difference")
    axes[1, 0].axis("off")
    fig.colorbar(abs_plot, ax=axes[1, 0], fraction=0.046, pad=0.04, label="absolute intensity difference")

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
        f"HR-PNG tumor pixels: {tumor_pixels}\n"
        f"HR-PNG tumor-only Pearson correlation: {correlation:.5f}",
        va="top",
        family="monospace",
        fontsize=12,
    )

    fig.suptitle("Pixel-wise HR/LR comparison", fontsize=16)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"shape={first_u8.shape}")
    print(f"hr_png_tumor_pixels={tumor_pixels}")
    print(f"hr_png_tumor_only_correlation={correlation:.8f}")
    print(f"output={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
