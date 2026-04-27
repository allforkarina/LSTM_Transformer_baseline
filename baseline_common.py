from __future__ import annotations

"""Shared utilities for sequence-level baseline training and evaluation."""

import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import BaselineLSTMTransformer, BaselineModelConfig


PCK_PIXEL_THRESHOLDS = (10.0, 20.0, 30.0)
COCO_BONES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class PoseSequenceLoss(nn.Module):
    """SmoothL1 pose loss with optional bone and temporal consistency terms."""

    def __init__(self, beta: float, bone_weight: float, temporal_weight: float) -> None:
        super().__init__()
        self.pose_loss = nn.SmoothL1Loss(beta=beta)
        self.bone_loss = nn.SmoothL1Loss(beta=beta)
        self.temporal_loss = nn.SmoothL1Loss(beta=beta)
        self.bone_weight = float(bone_weight)
        self.temporal_weight = float(temporal_weight)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        pose = self.pose_loss(predictions, targets)
        bone = self._bone_length_loss(predictions, targets)
        temporal = self._temporal_delta_loss(predictions, targets)
        total = pose + self.bone_weight * bone + self.temporal_weight * temporal
        return {
            "loss": total,
            "pose_loss": pose,
            "bone_loss": bone,
            "temp_loss": temporal,
        }

    def _bone_length_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bone_indices = torch.as_tensor(COCO_BONES, device=predictions.device, dtype=torch.long)
        pred_start = predictions.index_select(dim=2, index=bone_indices[:, 0])
        pred_end = predictions.index_select(dim=2, index=bone_indices[:, 1])
        target_start = targets.index_select(dim=2, index=bone_indices[:, 0])
        target_end = targets.index_select(dim=2, index=bone_indices[:, 1])
        pred_lengths = torch.linalg.vector_norm(pred_start - pred_end, dim=-1)
        target_lengths = torch.linalg.vector_norm(target_start - target_end, dim=-1)
        return self.bone_loss(pred_lengths, target_lengths)

    def _temporal_delta_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape[1] < 2:
            return predictions.new_zeros(())
        pred_delta = predictions[:, 1:] - predictions[:, :-1]
        target_delta = targets[:, 1:] - targets[:, :-1]
        return self.temporal_loss(pred_delta, target_delta)


def build_model(model_config: BaselineModelConfig | None = None) -> BaselineLSTMTransformer:
    """Construct the sequence baseline model."""

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


def get_dataset_scales(dataset: Dataset) -> tuple[float, float]:
    """Read the axis-wise keypoint scales stored in one HDF5-backed dataset."""

    return float(dataset.keypoint_x_scale), float(dataset.keypoint_y_scale)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one sequence batch onto the target device.

    The model input is ``B x T x 2 x 3 x 114 x 10``. Targets keep the existing
    HDF5 normalization convention and therefore stay in the ``[0, 1]`` range.
    """

    csi_amplitude = batch["csi_amplitude"].to(device=device, dtype=torch.float32)
    csi_phase_cos = batch["csi_phase_cos"].to(device=device, dtype=torch.float32)
    csi_window = torch.stack((csi_amplitude, csi_phase_cos), dim=2)
    targets = batch["keypoints"].to(device=device, dtype=torch.float32)
    return csi_window, targets


def normalized_to_pixels(keypoints: torch.Tensor, x_scale: float, y_scale: float) -> torch.Tensor:
    """Restore ``[0, 1]`` normalized coordinates to pixel coordinates."""

    scale = torch.tensor((x_scale, y_scale), device=keypoints.device, dtype=keypoints.dtype)
    return keypoints * scale.view(1, 1, 1, 2)


def compute_batch_statistics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    x_scale: float,
    y_scale: float,
) -> Dict[str, float]:
    """Compute pixel-space MPJPE and PCK for one sequence batch."""

    prediction_pixels = normalized_to_pixels(predictions, x_scale=x_scale, y_scale=y_scale)
    target_pixels = normalized_to_pixels(targets, x_scale=x_scale, y_scale=y_scale)
    distances_pixels = torch.linalg.vector_norm(prediction_pixels - target_pixels, dim=-1)
    statistics = {
        "distance_sum_pixels": float(distances_pixels.sum().item()),
        "num_keypoints": float(distances_pixels.numel()),
    }
    for threshold in PCK_PIXEL_THRESHOLDS:
        statistics[f"pck_{int(threshold)}px_correct"] = float((distances_pixels <= threshold).sum().item())
    return statistics


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: PoseSequenceLoss,
    x_scale: float,
    y_scale: float,
    phase_name: str,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
) -> Dict[str, float]:
    """Run one train/eval pass and aggregate sequence-level metrics."""

    is_training = optimizer is not None
    model.train(mode=is_training)
    total_samples = 0
    loss_sums = {"loss": 0.0, "pose_loss": 0.0, "bone_loss": 0.0, "temp_loss": 0.0}
    total_distance_sum_pixels = 0.0
    total_keypoints = 0.0
    pck_correct = {threshold: 0.0 for threshold in PCK_PIXEL_THRESHOLDS}

    progress_bar = tqdm(dataloader, desc=phase_name, leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        csi_window, targets = move_batch_to_device(batch, device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions = model(csi_window)
            loss_parts = criterion(predictions, targets)

        if is_training:
            loss_parts["loss"].backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        batch_size = int(targets.shape[0])
        total_samples += batch_size
        for name in loss_sums:
            loss_sums[name] += float(loss_parts[name].detach().item()) * batch_size

        batch_stats = compute_batch_statistics(predictions.detach(), targets.detach(), x_scale=x_scale, y_scale=y_scale)
        total_distance_sum_pixels += batch_stats["distance_sum_pixels"]
        total_keypoints += batch_stats["num_keypoints"]
        for threshold in PCK_PIXEL_THRESHOLDS:
            pck_correct[threshold] += batch_stats[f"pck_{int(threshold)}px_correct"]
        progress_bar.set_postfix(
            loss=f"{loss_parts['loss'].item():.4f}",
            mpjpe=f"{batch_stats['distance_sum_pixels'] / batch_stats['num_keypoints']:.2f}",
        )

    if total_samples == 0 or total_keypoints == 0:
        raise ValueError("The dataloader produced no sequence windows.")

    metrics = {name: value / total_samples for name, value in loss_sums.items()}
    metrics["mpjpe"] = total_distance_sum_pixels / total_keypoints
    for threshold in PCK_PIXEL_THRESHOLDS:
        metrics[f"pck_{int(threshold)}px"] = pck_correct[threshold] / total_keypoints
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
