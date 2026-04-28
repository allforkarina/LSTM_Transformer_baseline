from __future__ import annotations

"""Pose visualization utilities for CSI2Pose checkpoint inspection."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np
import torch

from dataloader import _decode_string
from metrics import denormalize_keypoints_tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COCO_SKELETON = (
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
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)
SKELETON_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#ff7f0e",
    "#2ca02c",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#7f7f7f",
    "#d62728",
    "#d62728",
    "#17becf",
    "#17becf",
    "#bcbd22",
    "#bcbd22",
    "#e377c2",
    "#e377c2",
)


@dataclass(frozen=True)
class VisualizationFrame:
    """One selected sequence-window middle frame for visualization."""

    dataset_index: int
    frame_index: int
    action: str
    environment: str
    sample: str
    frame_id: str

    @property
    def output_name(self) -> str:
        return f"{self.action}_{self.environment}_{self.sample}_{self.frame_id}.png"


def select_one_frame_per_action_env_sample(dataset: Any, seed: int) -> list[VisualizationFrame]:
    """Randomly select one window-middle frame per action/environment/sample group."""

    grouped: dict[tuple[str, str, str], list[VisualizationFrame]] = {}
    with h5py.File(dataset.dataset_root, "r") as h5_file:
        actions = np.asarray(h5_file["action"])
        samples = np.asarray(h5_file["sample"])
        environments = np.asarray(h5_file["environment"])
        frame_ids = np.asarray(h5_file["frame_id"])
        for dataset_index, window_indices in enumerate(dataset.windows):
            frame_index = int(window_indices[len(window_indices) // 2])
            action = _decode_string(actions[frame_index])
            sample = _decode_string(samples[frame_index])
            environment = _decode_string(environments[frame_index])
            frame_id = _decode_string(frame_ids[frame_index])
            frame = VisualizationFrame(
                dataset_index=dataset_index,
                frame_index=frame_index,
                action=action,
                environment=environment,
                sample=sample,
                frame_id=frame_id,
            )
            grouped.setdefault((action, environment, sample), []).append(frame)

    rng = random.Random(seed)
    return [
        rng.choice(grouped[group_key])
        for group_key in sorted(grouped)
    ]


def _draw_colored_skeleton(ax: Any, keypoints: np.ndarray, title: str) -> None:
    ax.scatter(keypoints[:, 0], keypoints[:, 1], s=16, color="#222222", zorder=3)
    for (start, end), color in zip(COCO_SKELETON, SKELETON_COLORS):
        ax.plot(
            [keypoints[start, 0], keypoints[end, 0]],
            [keypoints[start, 1], keypoints[end, 1]],
            color=color,
            linewidth=2.0,
            zorder=2,
        )
    ax.set_title(title)


def _draw_single_color_skeleton(ax: Any, keypoints: np.ndarray, color: str, label: str) -> None:
    ax.scatter(keypoints[:, 0], keypoints[:, 1], s=14, color=color, label=label, zorder=3)
    for start, end in COCO_SKELETON:
        ax.plot(
            [keypoints[start, 0], keypoints[end, 0]],
            [keypoints[start, 1], keypoints[end, 1]],
            color=color,
            linewidth=1.8,
            zorder=2,
        )


def _set_pose_axes(ax: Any, keypoint_sets: Iterable[np.ndarray], title: str) -> None:
    stacked = np.concatenate(list(keypoint_sets), axis=0)
    x_min, y_min = np.min(stacked, axis=0)
    x_max, y_max = np.max(stacked, axis=0)
    padding = max(float(x_max - x_min), float(y_max - y_min), 1.0) * 0.12
    ax.set_xlim(float(x_min - padding), float(x_max + padding))
    ax.set_ylim(float(y_max + padding), float(y_min - padding))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_title(title)


def save_pose_comparison(
    gt_keypoints: torch.Tensor,
    predicted_keypoints: torch.Tensor,
    x_scale: float,
    y_scale: float,
    output_path: str | Path,
) -> None:
    """Save a 3x1 GT/prediction pose comparison in pixel coordinates."""

    gt_pixels = denormalize_keypoints_tensor(gt_keypoints.detach().cpu(), x_scale=x_scale, y_scale=y_scale).numpy()
    pred_pixels = denormalize_keypoints_tensor(
        predicted_keypoints.detach().cpu(),
        x_scale=x_scale,
        y_scale=y_scale,
    ).numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7, 15), constrained_layout=True)
    _draw_colored_skeleton(axes[0], gt_pixels, "GT")
    _draw_colored_skeleton(axes[1], pred_pixels, "Prediction")
    _draw_single_color_skeleton(axes[2], gt_pixels, color="#1f77b4", label="GT")
    _draw_single_color_skeleton(axes[2], pred_pixels, color="#d62728", label="Prediction")
    for joint_index in range(gt_pixels.shape[0]):
        axes[2].plot(
            [gt_pixels[joint_index, 0], pred_pixels[joint_index, 0]],
            [gt_pixels[joint_index, 1], pred_pixels[joint_index, 1]],
            color="#555555",
            linewidth=0.6,
            alpha=0.75,
            zorder=1,
        )
    axes[2].legend(loc="upper right")

    for axis, title in zip(axes, ("GT", "Prediction", "GT vs Prediction")):
        _set_pose_axes(axis, (gt_pixels, pred_pixels), title)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_visualization_samples(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    output_dir: str | Path,
    seed: int,
) -> list[Path]:
    """Run model inference and save one visualization per action/environment/sample group."""

    model.eval()
    output_dir = Path(output_dir)
    selected_frames = select_one_frame_per_action_env_sample(dataset, seed=seed)
    saved_paths: list[Path] = []
    middle_index = dataset.window_size // 2

    for frame in selected_frames:
        sample = dataset[frame.dataset_index]
        csi_amplitude = torch.as_tensor(sample["csi_amplitude"], dtype=torch.float32, device=device).unsqueeze(0)
        csi_phase_cos = torch.as_tensor(sample["csi_phase_cos"], dtype=torch.float32, device=device).unsqueeze(0)
        keypoints = torch.as_tensor(sample["keypoints"], dtype=torch.float32)
        outputs = model(csi_amplitude, csi_phase_cos)
        output_path = output_dir / frame.output_name
        save_pose_comparison(
            gt_keypoints=keypoints[middle_index],
            predicted_keypoints=outputs["keypoints"][0, middle_index].detach().cpu(),
            x_scale=dataset.keypoint_x_scale,
            y_scale=dataset.keypoint_y_scale,
            output_path=output_path,
        )
        saved_paths.append(output_path)

    return saved_paths
