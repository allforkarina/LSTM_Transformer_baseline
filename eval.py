from __future__ import annotations

"""Evaluation entrypoint for the sequence-level MM-Fi CSI pose baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline_common import (
    PCK_PIXEL_THRESHOLDS,
    PoseSequenceLoss,
    build_dataloader,
    build_model,
    ensure_output_dir,
    get_dataset_scales,
    model_config_from_dict,
    move_batch_to_device,
    normalized_to_pixels,
    resolve_device,
    run_epoch,
)
from dataloader import MMFiPoseSequenceDataset


COCO_KEYPOINT_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:orange",
    "tab:orange",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:green",
    "tab:red",
    "tab:green",
    "tab:red",
    "tab:green",
    "tab:red",
    "tab:green",
    "tab:red",
    "tab:green",
    "tab:red",
)
COCO_SKELETON = (
    (0, 1, "tab:blue"),
    (0, 2, "tab:blue"),
    (1, 3, "tab:blue"),
    (2, 4, "tab:blue"),
    (0, 5, "tab:blue"),
    (0, 6, "tab:blue"),
    (5, 6, "tab:blue"),
    (5, 7, "tab:green"),
    (7, 9, "tab:green"),
    (6, 8, "tab:red"),
    (8, 10, "tab:red"),
    (5, 11, "tab:green"),
    (6, 12, "tab:red"),
    (11, 12, "tab:blue"),
    (11, 13, "tab:green"),
    (13, 15, "tab:green"),
    (12, 14, "tab:red"),
    (14, 16, "tab:red"),
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for baseline evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate the MM-Fi sequence CSI pose baseline")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to one HDF5 dataset file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to one saved training checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for evaluation outputs")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"), help="Dataset split")
    parser.add_argument(
        "--split-scheme",
        type=str,
        default=None,
        choices=("action_env", "frame_random"),
        help="Override the split scheme stored in the checkpoint config",
    )
    parser.add_argument("--window-size", type=int, default=None, help="Override checkpoint window size")
    parser.add_argument("--window-stride", type=int, default=None, help="Override checkpoint window stride")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of PyTorch dataloader workers")
    parser.add_argument("--device", type=str, default="auto", help="Runtime device, for example auto, cuda, or cpu")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""

    return build_arg_parser().parse_args(argv)


def _normalize_heatmap(csi_heatmap: np.ndarray) -> np.ndarray:
    """Normalize one CSI heatmap into the ``[0, 1]`` range for plotting."""

    heatmap = np.asarray(csi_heatmap, dtype=np.float32)
    finite_mask = np.isfinite(heatmap)
    if not np.any(finite_mask):
        return np.zeros_like(heatmap, dtype=np.float32)

    finite_values = heatmap[finite_mask]
    heatmap_min = float(finite_values.min())
    heatmap_max = float(finite_values.max())
    if heatmap_max <= heatmap_min:
        normalized = np.zeros_like(heatmap, dtype=np.float32)
        normalized[finite_mask] = 1.0
        return normalized

    normalized = np.zeros_like(heatmap, dtype=np.float32)
    normalized[finite_mask] = (heatmap[finite_mask] - heatmap_min) / (heatmap_max - heatmap_min)
    return normalized


def _build_csi_heatmap(csi_amplitude: np.ndarray) -> np.ndarray:
    """Stack three antenna channels vertically into one ``342 x 10`` heatmap."""

    csi_amplitude = np.asarray(csi_amplitude, dtype=np.float32)
    if csi_amplitude.shape != (3, 114, 10):
        raise ValueError(f"Expected one CSI amplitude tensor with shape (3, 114, 10), got {csi_amplitude.shape}")
    return csi_amplitude.reshape(3 * 114, 10)


def _compute_pose_limits(target_pose: np.ndarray, prediction_pose: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute one shared plotting window for target and predicted poses."""

    combined_pose = np.concatenate((target_pose, prediction_pose), axis=0)
    finite_mask = np.isfinite(combined_pose).all(axis=1)
    if not np.any(finite_mask):
        return (-1.0, 1.0), (-1.0, 1.0)

    finite_pose = combined_pose[finite_mask]
    x_min = float(finite_pose[:, 0].min())
    x_max = float(finite_pose[:, 0].max())
    y_min = float(finite_pose[:, 1].min())
    y_max = float(finite_pose[:, 1].max())
    x_padding = max((x_max - x_min) * 0.05, 1.0)
    y_padding = max((y_max - y_min) * 0.05, 1.0)
    return (x_min - x_padding, x_max + x_padding), (y_min - y_padding, y_max + y_padding)


def _draw_pose(axis, pose: np.ndarray, title: str, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    """Draw one COCO-17 pose skeleton on the provided matplotlib axis."""

    pose = np.asarray(pose, dtype=np.float32)
    for start_index, end_index, color in COCO_SKELETON:
        if np.any(np.isnan(pose[start_index])) or np.any(np.isnan(pose[end_index])):
            continue
        axis.plot(
            [pose[start_index, 0], pose[end_index, 0]],
            [pose[start_index, 1], pose[end_index, 1]],
            color=color,
            linewidth=2.5,
            solid_capstyle="round",
        )

    for keypoint_index, color in enumerate(COCO_KEYPOINT_COLORS):
        if np.any(np.isnan(pose[keypoint_index])):
            continue
        axis.scatter(
            pose[keypoint_index, 0],
            pose[keypoint_index, 1],
            s=42,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.invert_yaxis()
    axis.axis("off")


def _expected_visualization_groups(dataset: MMFiPoseSequenceDataset) -> list[tuple[str, str]]:
    """Return sorted ``(action, environment)`` pairs covered by sequence windows."""

    groups: set[tuple[str, str]] = set()
    h5_file = dataset.frame_dataset._get_h5_file()
    for window in dataset.windows:
        middle_index = window[len(window) // 2]
        action = h5_file["action"][middle_index]
        environment = h5_file["environment"][middle_index]
        action = action.decode("utf-8") if isinstance(action, bytes) else str(action)
        environment = environment.decode("utf-8") if isinstance(environment, bytes) else str(environment)
        groups.add((action, environment))
    return sorted(groups)


def _save_visualizations(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    x_scale: float,
    y_scale: float,
    dataset: MMFiPoseSequenceDataset,
) -> dict[str, object]:
    """Save one middle-frame CSI-plus-skeleton figure per ``(action, environment)``."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on optional runtime package.
        raise ImportError("matplotlib is required to save evaluation visualizations.") from error

    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    expected_groups = _expected_visualization_groups(dataset)
    saved_groups: set[tuple[str, str]] = set()
    selected_groups: list[str] = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            csi_window, targets = move_batch_to_device(batch, device)
            predictions = model(csi_window)
            prediction_pixels = normalized_to_pixels(predictions, x_scale, y_scale).detach().cpu().numpy()
            target_pixels = normalized_to_pixels(targets, x_scale, y_scale).detach().cpu().numpy()
            heatmaps = batch["csi_amplitude"].detach().cpu().numpy()
            middle = heatmaps.shape[1] // 2

            for index in range(prediction_pixels.shape[0]):
                group = (str(batch["action"][index]), str(batch["environment"][index]))
                if group in saved_groups:
                    continue

                x_limits, y_limits = _compute_pose_limits(target_pixels[index, middle], prediction_pixels[index, middle])
                normalized_heatmap = _normalize_heatmap(_build_csi_heatmap(heatmaps[index, middle]))
                figure, axes = plt.subplots(
                    nrows=3,
                    ncols=1,
                    figsize=(6, 10),
                    dpi=200,
                    gridspec_kw={"height_ratios": (1.1, 1.0, 1.0)},
                )
                heatmap_axis, target_axis, prediction_axis = axes
                image = heatmap_axis.imshow(normalized_heatmap, aspect="auto", origin="lower", cmap="jet")
                heatmap_axis.set_title(
                    f"{batch['action'][index]} {batch['sample'][index]} "
                    f"{batch['environment'][index]} {batch['frame_id'][index]}"
                )
                heatmap_axis.set_xlabel("CSI time shot")
                heatmap_axis.set_ylabel("Antenna/Subcarrier")
                figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04, label="Normalized amplitude")
                _draw_pose(target_axis, target_pixels[index, middle], "Ground Truth", x_limits, y_limits)
                _draw_pose(prediction_axis, prediction_pixels[index, middle], "Prediction", x_limits, y_limits)

                file_name = (
                    f"{batch['action'][index]}_{batch['environment'][index]}_"
                    f"{batch['sample'][index]}_{batch['frame_id'][index]}.png"
                )
                figure.tight_layout()
                figure.savefig(visualization_dir / file_name, dpi=150)
                plt.close(figure)
                saved_groups.add(group)
                selected_groups.append(f"{group[0]}/{group[1]}")
                if len(saved_groups) == len(expected_groups):
                    break

            if len(saved_groups) == len(expected_groups):
                break

    missing_groups = [
        f"{action}/{environment}"
        for action, environment in expected_groups
        if (action, environment) not in saved_groups
    ]
    return {
        "visualization_mode": "per_action_environment_middle_frame",
        "num_visualizations_saved": len(saved_groups),
        "selected_groups": selected_groups,
        "missing_groups": missing_groups,
    }


def main(argv: list[str] | None = None) -> dict[str, float]:
    """Evaluate one saved sequence baseline checkpoint."""

    args = parse_args(argv)
    device = resolve_device(args.device)
    output_dir = ensure_output_dir(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    split_scheme = args.split_scheme or checkpoint_config.get("split_scheme", "action_env")
    window_size = args.window_size or int(checkpoint_config.get("window_size", 16))
    window_stride = args.window_stride or int(checkpoint_config.get("window_stride", 4))
    model_config = model_config_from_dict(checkpoint_config["model"])

    dataset = MMFiPoseSequenceDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        split_scheme=split_scheme,
        window_size=window_size,
        window_stride=window_stride,
    )
    dataloader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    x_scale, y_scale = get_dataset_scales(dataset)

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = PoseSequenceLoss(
        beta=float(checkpoint_config.get("smooth_l1_beta", 0.02)),
        bone_weight=float(checkpoint_config.get("bone_loss_weight", 0.1)),
        temporal_weight=float(checkpoint_config.get("temporal_loss_weight", 0.05)),
    )
    metrics = run_epoch(
        model,
        dataloader,
        device=device,
        criterion=criterion,
        x_scale=x_scale,
        y_scale=y_scale,
        phase_name=args.split,
    )
    metrics_payload = {
        "dataset_root": args.dataset_root,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "split_scheme": split_scheme,
        "window_size": window_size,
        "window_stride": window_stride,
        "num_windows": len(dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        "metrics": metrics,
        "pck_pixel_thresholds": list(PCK_PIXEL_THRESHOLDS),
    }
    metrics_payload["visualizations"] = _save_visualizations(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        x_scale=x_scale,
        y_scale=y_scale,
        dataset=dataset,
    )
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    main()
