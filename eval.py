from __future__ import annotations

"""Evaluation entrypoint for the frame-level MM-Fi LSTM + Transformer baseline."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from baseline_common import (
    PCK_THRESHOLDS,
    build_dataloader,
    build_model,
    ensure_output_dir,
    get_dataset_scales,
    model_config_from_dict,
    move_batch_to_device,
    resolve_device,
    run_epoch,
)
from dataloader import MMFiPoseDataset


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for baseline evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate the MM-Fi LSTM + Transformer baseline")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to one HDF5 dataset file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to one saved training checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for evaluation outputs")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--split-scheme",
        type=str,
        default=None,
        choices=("action_env", "frame_random"),
        help="Override the split scheme stored in the checkpoint config",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of PyTorch dataloader workers")
    parser.add_argument("--device", type=str, default="auto", help="Runtime device, for example auto, cuda, or cpu")
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=8,
        help="Number of prediction-vs-target plots to save",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""

    return build_arg_parser().parse_args(argv)


def _save_visualizations(
    model: nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    x_scale: float,
    y_scale: float,
    num_visualizations: int,
) -> None:
    """Save a small set of prediction-vs-target keypoint plots.

    The saved figures are intentionally simple. They exist to verify that predicted
    skeleton geometry is plausible, not to build a polished visualization system.
    """

    if num_visualizations <= 0:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on optional runtime package.
        raise ImportError(
            "matplotlib is required to save evaluation visualizations. "
            "Install matplotlib or set --num-visualizations 0."
        ) from error

    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    scale = torch.tensor((x_scale, y_scale), device=device, dtype=torch.float32).view(1, 1, 2)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            csi_amplitude, csi_phase_cos, targets = move_batch_to_device(batch, device)
            predictions = model(csi_amplitude, csi_phase_cos)
            prediction_pixels = (predictions * scale).detach().cpu().numpy()
            target_pixels = (targets * scale).detach().cpu().numpy()
            actions = batch["action"]
            samples = batch["sample"]
            environments = batch["environment"]
            frame_ids = batch["frame_id"]

            for index in range(prediction_pixels.shape[0]):
                if saved >= num_visualizations:
                    return

                figure, axis = plt.subplots(figsize=(5, 5))
                axis.scatter(target_pixels[index, :, 0], target_pixels[index, :, 1], c="tab:green", label="target")
                axis.scatter(prediction_pixels[index, :, 0], prediction_pixels[index, :, 1], c="tab:red", label="prediction")
                axis.set_title(
                    f"{actions[index]} {samples[index]} {environments[index]} {frame_ids[index]}"
                )
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.invert_yaxis()
                axis.legend(loc="best")
                axis.set_aspect("equal", adjustable="box")
                file_name = (
                    f"{actions[index]}_{samples[index]}_{environments[index]}_{frame_ids[index]}.png"
                )
                figure.tight_layout()
                figure.savefig(visualization_dir / file_name, dpi=150)
                plt.close(figure)
                saved += 1


def main(argv: list[str] | None = None) -> dict[str, float]:
    """Evaluate one saved baseline checkpoint on a requested split."""

    args = parse_args(argv)
    device = resolve_device(args.device)
    output_dir = ensure_output_dir(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    split_scheme = args.split_scheme or checkpoint_config.get("split_scheme", "action_env")
    model_config = model_config_from_dict(checkpoint_config["model"])

    # Evaluation reuses the same split statistics that were written into the HDF5 file
    # so metrics stay consistent with training-time normalization.
    dataset = MMFiPoseDataset(dataset_root=args.dataset_root, split=args.split, split_scheme=split_scheme)
    dataloader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    x_scale, y_scale = get_dataset_scales(dataset)

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.SmoothL1Loss(beta=float(checkpoint_config.get("smooth_l1_beta", 0.02)))
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
        "batch_size": args.batch_size,
        "device": str(device),
        "metrics": metrics,
        "pck_thresholds": list(PCK_THRESHOLDS),
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    _save_visualizations(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        x_scale=x_scale,
        y_scale=y_scale,
        num_visualizations=args.num_visualizations,
    )
    return metrics


if __name__ == "__main__":
    main()
