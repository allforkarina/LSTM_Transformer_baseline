from __future__ import annotations

"""Shared utilities for baseline training and evaluation."""

import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from models import BaselineLSTMTransformer, BaselineModelConfig


PCK_THRESHOLDS = (0.05, 0.10, 0.20)


def build_model(model_config: BaselineModelConfig | None = None) -> BaselineLSTMTransformer:
    """Construct the baseline model from one config object."""

    return BaselineLSTMTransformer(config=model_config or BaselineModelConfig())


def model_config_to_dict(model_config: BaselineModelConfig) -> Dict[str, int | float]:
    """Convert the model config to a JSON-friendly dictionary."""

    return asdict(model_config)


def model_config_from_dict(config_dict: Dict[str, int | float]) -> BaselineModelConfig:
    """Restore the model config from a checkpoint dictionary."""

    return BaselineModelConfig(**config_dict)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create one output directory and return it as a path."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_subset_dataset(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    """Return a deterministic subset when the caller requests a smaller split view."""

    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def get_dataset_scales(dataset: Dataset) -> tuple[float, float]:
    """Read keypoint normalization scales from a dataset or subset wrapper."""

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    return float(base_dataset.keypoint_x_scale), float(base_dataset.keypoint_y_scale)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the tensor fields used by the baseline and move them onto one device."""

    csi_amplitude = batch["csi_amplitude"].to(device=device, dtype=torch.float32)
    csi_phase_cos = batch["csi_phase_cos"].to(device=device, dtype=torch.float32)
    keypoints = batch["keypoints"].to(device=device, dtype=torch.float32)
    return csi_amplitude, csi_phase_cos, keypoints


def compute_batch_statistics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    x_scale: float,
    y_scale: float,
) -> Dict[str, float]:
    """Compute one batch's pose metrics in normalized and pixel spaces."""

    scale = torch.tensor((x_scale, y_scale), device=predictions.device, dtype=predictions.dtype).view(1, 1, 2)
    prediction_pixels = predictions * scale
    target_pixels = targets * scale
    distances_pixels = torch.linalg.vector_norm(prediction_pixels - target_pixels, dim=-1)
    distances_normalized = torch.linalg.vector_norm(predictions - targets, dim=-1)
    statistics = {
        "distance_sum_pixels": float(distances_pixels.sum().item()),
        "num_keypoints": float(distances_pixels.numel()),
    }
    for threshold in PCK_THRESHOLDS:
        statistics[f"pck_{threshold:.2f}_correct"] = float((distances_normalized <= threshold).sum().item())
    return statistics


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    x_scale: float,
    y_scale: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    """Run one training or evaluation epoch and return aggregated metrics."""

    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_samples = 0
    total_distance_sum_pixels = 0.0
    total_keypoints = 0.0
    pck_correct = {threshold: 0.0 for threshold in PCK_THRESHOLDS}

    for batch in dataloader:
        csi_amplitude, csi_phase_cos, targets = move_batch_to_device(batch, device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions = model(csi_amplitude, csi_phase_cos)
            loss = criterion(predictions, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        batch_stats = compute_batch_statistics(predictions.detach(), targets.detach(), x_scale=x_scale, y_scale=y_scale)
        total_distance_sum_pixels += batch_stats["distance_sum_pixels"]
        total_keypoints += batch_stats["num_keypoints"]
        for threshold in PCK_THRESHOLDS:
            pck_correct[threshold] += batch_stats[f"pck_{threshold:.2f}_correct"]

    if total_samples == 0 or total_keypoints == 0:
        raise ValueError("The dataloader produced no samples.")

    metrics = {
        "loss": total_loss / total_samples,
        "mpjpe": total_distance_sum_pixels / total_keypoints,
    }
    for threshold in PCK_THRESHOLDS:
        metrics[f"pck_{threshold:.2f}"] = pck_correct[threshold] / total_keypoints
    return metrics


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a standard PyTorch dataloader for one dataset view."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested runtime device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_random_seed(seed: int) -> None:
    """Set runtime seeds for reproducible baseline experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
