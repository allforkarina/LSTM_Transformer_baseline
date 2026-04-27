from __future__ import annotations

"""Pose metrics for normalized COCO17 keypoint predictions."""

from typing import Iterable

import torch

from models import COCO_KEYPOINT_NAMES


DEFAULT_PCK_THRESHOLDS = (5.0, 10.0, 20.0, 50.0)


def denormalize_keypoints_tensor(
    keypoints: torch.Tensor,
    x_scale: float,
    y_scale: float,
) -> torch.Tensor:
    """Restore normalized coordinates to the original pixel-coordinate scale."""

    scale = torch.tensor([x_scale, y_scale], dtype=keypoints.dtype, device=keypoints.device)
    return keypoints * scale


def compute_pck(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    x_scale: float,
    y_scale: float,
    thresholds: Iterable[float] = DEFAULT_PCK_THRESHOLDS,
) -> dict[str, float | dict[str, dict[str, float]]]:
    """Compute overall and per-joint PCK in pixel space.

    Args:
        predictions: Normalized predicted keypoints shaped ``... x 17 x 2``.
        targets: Normalized target keypoints shaped ``... x 17 x 2``.
        x_scale: Pixel scale used to restore normalized x coordinates.
        y_scale: Pixel scale used to restore normalized y coordinates.
        thresholds: Pixel-distance thresholds for PCK.
    """

    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction and target shapes differ: {predictions.shape} vs {targets.shape}")
    if predictions.shape[-2:] != (len(COCO_KEYPOINT_NAMES), 2):
        raise ValueError(f"Expected trailing shape 17 x 2, got {predictions.shape[-2:]}")

    pred_pixels = denormalize_keypoints_tensor(predictions, x_scale=x_scale, y_scale=y_scale)
    target_pixels = denormalize_keypoints_tensor(targets, x_scale=x_scale, y_scale=y_scale)
    distances = torch.linalg.vector_norm(pred_pixels - target_pixels, dim=-1)

    result: dict[str, float | dict[str, dict[str, float]]] = {}
    per_joint: dict[str, dict[str, float]] = {name: {} for name in COCO_KEYPOINT_NAMES}
    for threshold in thresholds:
        metric_name = f"pck@{int(threshold)}"
        correct = distances <= float(threshold)
        result[metric_name] = float(correct.float().mean().item())
        joint_scores = correct.float().reshape(-1, len(COCO_KEYPOINT_NAMES)).mean(dim=0)
        for joint_index, joint_name in enumerate(COCO_KEYPOINT_NAMES):
            per_joint[joint_name][metric_name] = float(joint_scores[joint_index].item())

    result["per_joint"] = per_joint
    return result


class PCKAccumulator:
    """Accumulate predictions across batches before computing PCK once."""

    def __init__(self, x_scale: float, y_scale: float, thresholds: Iterable[float] = DEFAULT_PCK_THRESHOLDS) -> None:
        self.x_scale = float(x_scale)
        self.y_scale = float(y_scale)
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self._predictions.append(predictions.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> dict[str, float | dict[str, dict[str, float]]]:
        if not self._predictions:
            raise ValueError("PCKAccumulator has no samples")
        predictions = torch.cat([item.reshape(-1, len(COCO_KEYPOINT_NAMES), 2) for item in self._predictions], dim=0)
        targets = torch.cat([item.reshape(-1, len(COCO_KEYPOINT_NAMES), 2) for item in self._targets], dim=0)
        return compute_pck(
            predictions,
            targets,
            x_scale=self.x_scale,
            y_scale=self.y_scale,
            thresholds=self.thresholds,
        )
