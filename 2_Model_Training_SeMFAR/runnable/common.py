"""Shared, self-contained models and data utilities for runnable Part 2 experiments."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gray(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def discover_pairs(paired: Path) -> List[Tuple[str, Path, Path]]:
    records: List[Tuple[str, Path, Path]] = []
    for lr_path in sorted(paired.glob("*_LR.png")):
        hr_path = lr_path.with_name(lr_path.name.replace("_LR.png", "_HR.png"))
        if hr_path.is_file():
            records.append((lr_path.stem[:-3], lr_path, hr_path))
    return records


def load_split(path: Path, records: Sequence[Tuple[str, Path, Path]]) -> Dict[str, list]:
    split = json.loads(path.read_text(encoding="utf-8-sig"))
    by_name = {record[0]: record for record in records}
    required = ("train", "validation", "test")
    missing_sections = [key for key in required if key not in split]
    if missing_sections:
        raise ValueError(f"Split is missing sections: {missing_sections}")
    missing_names = [
        name for key in required for name in split[key] if name not in by_name
    ]
    if missing_names:
        raise ValueError(
            f"Split references {len(missing_names)} unavailable pairs; first: {missing_names[:5]}"
        )
    return split


def split_records(
    split: Dict[str, list], records: Sequence[Tuple[str, Path, Path]]
) -> Dict[str, list]:
    by_name = {record[0]: record for record in records}
    return {
        key: [by_name[name] for name in split[key]]
        for key in ("train", "validation", "test")
    }


class PairDataset(Dataset):
    def __init__(self, records: Sequence[Tuple[str, Path, Path]], augment: bool = False):
        self.records = list(records)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        name, lr_path, hr_path = self.records[index]
        lr, hr = load_gray(lr_path), load_gray(hr_path)
        if self.augment:
            turns = random.randrange(4)
            flip = random.random() < 0.5
            lr = torch.rot90(lr, turns, (1, 2))
            hr = torch.rot90(hr, turns, (1, 2))
            if flip:
                lr, hr = lr.flip(2), hr.flip(2)
        return lr, hr, name


class UnpairedDataset(Dataset):
    def __init__(self, files: Iterable[Path]):
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        return load_gray(self.files[index])


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64, scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.scale


class EDSR(nn.Module):
    def __init__(self, residual_scale: float = 0.1):
        super().__init__()
        self.head = nn.Conv2d(1, 64, 3, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(scale=residual_scale) for _ in range(8)]
        )
        self.body_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.tail = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.head(x)
        feature = feature + self.body_conv(self.blocks(feature))
        return x + self.tail(feature)


def checkpoint_state(checkpoint: dict) -> dict:
    for key in ("student", "model", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise KeyError("Checkpoint contains no student/model/state_dict parameters")


def charbonnier(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-3):
    return torch.sqrt((first - second).square() + eps * eps).mean()


def gradient_charbonnier(prediction: torch.Tensor, target: torch.Tensor):
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (charbonnier(pred_x, target_x) + charbonnier(pred_y, target_y))


def ssim(prediction: torch.Tensor, target: torch.Tensor):
    pool = F.avg_pool2d
    mean_p, mean_t = pool(prediction, 11, 1, 5), pool(target, 11, 1, 5)
    var_p = pool(prediction * prediction, 11, 1, 5) - mean_p.square()
    var_t = pool(target * target, 11, 1, 5) - mean_t.square()
    cov = pool(prediction * target, 11, 1, 5) - mean_p * mean_t
    score = ((2 * mean_p * mean_t + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mean_p.square() + mean_t.square() + 0.01**2)
        * (var_p + var_t + 0.03**2)
    )
    return score.mean()


def gaussian(image: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    radius = int(np.ceil(3 * sigma))
    axis = torch.arange(
        -radius, radius + 1, device=image.device, dtype=image.dtype
    )
    kernel_1d = torch.exp(-0.5 * (axis / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
    return F.conv2d(F.pad(image, (radius,) * 4, mode="reflect"), kernel)


def fixed_2d_position(size: int = 32, dim: int = 256) -> torch.Tensor:
    quarter = dim // 4
    y_coord, x_coord = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    frequency = torch.exp(-math.log(10000) * torch.arange(quarter) / quarter)
    x_embed = x_coord.flatten().float()[:, None] * frequency
    y_embed = y_coord.flatten().float()[:, None] * frequency
    return torch.cat(
        (torch.sin(x_embed), torch.cos(x_embed), torch.sin(y_embed), torch.cos(y_embed)),
        dim=1,
    )


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.norm1(x)
        x = x + self.attention(query, query, query, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ResidualPredictor(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 8, heads: int = 8, patch: int = 8):
        super().__init__()
        self.patch = patch
        self.lr_conv = nn.Conv2d(1, 64, 3, padding=1)
        self.coarse_conv = nn.Conv2d(1, 64, 3, padding=1)
        self.lr_patch = nn.Conv2d(64, dim, patch, stride=patch)
        self.coarse_patch = nn.Conv2d(64, dim, patch, stride=patch)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.register_buffer("position", fixed_2d_position(256 // patch, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, 4) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch * patch)

    def forward(self, lr: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        query = self.lr_patch(self.lr_conv(lr)).flatten(2).transpose(1, 2)
        context = self.coarse_patch(self.coarse_conv(coarse)).flatten(2).transpose(1, 2)
        x = self.cross(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )[0]
        x = x + self.position[None].to(x.dtype)
        for block in self.blocks:
            x = block(x)
        patch = self.patch
        batch = x.shape[0]
        return (
            self.out(self.norm(x))
            .view(batch, 32, 32, patch, patch)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, 1, 256, 256)
        )


def segment_tumor_from_hr(hr_u8: np.ndarray) -> np.ndarray:
    height, width = hr_u8.shape
    blurred = cv2.GaussianBlur(hr_u8, (0, 0), 3.0)
    _, dark = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    margin_x = max(5, round(width * 0.035))
    margin_y = max(5, round(height * 0.035))
    dark[:margin_y, :] = 0
    dark[-margin_y:, :] = 0
    dark[:, :margin_x] = 0
    dark[:, -margin_x:] = 0
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    best_index, best_score = None, -np.inf
    image_area = float(height * width)
    for index in range(1, count):
        x, y, box_w, box_h, area = stats[index]
        fraction = float(area) / image_area
        if not 0.008 <= fraction <= 0.65:
            continue
        center_x, center_y = centroids[index]
        distance = np.hypot(
            (center_x - width / 2) / (width / 2),
            (center_y - height / 2) / (height / 2),
        )
        component = labels == index
        darkness = (255.0 - float(np.mean(hr_u8[component]))) / 255.0
        thickness = min(box_w / width, box_h / height)
        score = (
            np.log1p(float(area))
            * max(darkness, 1e-3)
            * max(thickness, 1e-3)
            * np.exp(-1.5 * distance)
        )
        if score > best_score:
            best_index, best_score = index, score
    if best_index is None:
        return np.ones_like(hr_u8, dtype=bool)
    return labels == best_index


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    mse = float(np.mean((prediction - target) ** 2))
    pred_tensor = torch.from_numpy(prediction)[None, None]
    target_tensor = torch.from_numpy(target)[None, None]
    target_u8 = np.clip(np.rint(target * 255), 0, 255).astype(np.uint8)
    mask = segment_tumor_from_hr(target_u8)
    roi_mse = float(np.mean((prediction[mask] - target[mask]) ** 2))
    return {
        "psnr_db": float(-10 * np.log10(max(mse, eps))),
        "ssim": float(ssim(pred_tensor, target_tensor)),
        "tumor_roi_psnr_db": float(-10 * np.log10(max(roi_mse, eps))),
        "tumor_pixels": int(mask.sum()),
    }


def aggregate_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, dict]:
    output: Dict[str, dict] = {"n": len(rows)}
    if not rows:
        return output
    for key in rows[0]:
        if key == "file" or key == "tumor_pixels":
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return output

