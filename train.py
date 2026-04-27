from __future__ import annotations

"""Train CSI2Pose v1 on windowed MM-Fi HDF5 data."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataloader import DEFAULT_SPLIT_SCHEME, MMFiPoseSequenceDataset
from metrics import PCKAccumulator
from models import CSI2PoseModel


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CSI2Pose heatmap baseline")
    parser.add_argument("--config", type=str, default="configs/csi2pose_tcn.yaml", help="YAML config path")
    parser.add_argument("--dataset-root", type=str, required=True, help="Packed MM-Fi HDF5 dataset path")
    parser.add_argument("--run-dir", type=str, default=None, help="Override output directory for checkpoints/metrics")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the experiment YAML config used by server-side training."""

    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def set_seed(seed: int) -> None:
    """Seed common random generators for reproducible lightweight baselines."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sequence_loader(
    dataset_root: str | Path,
    split: str,
    config: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader over contiguous CSI windows."""

    data_config = config["data"]
    train_config = config["train"]
    dataset = MMFiPoseSequenceDataset(
        dataset_root=dataset_root,
        split=split,
        split_scheme=data_config.get("split_scheme", DEFAULT_SPLIT_SCHEME),
        window_size=int(data_config["window_size"]),
        window_stride=int(data_config["window_stride"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_config["num_workers"]),
        pin_memory=bool(train_config["pin_memory"]),
    )


def build_model(config: dict[str, Any]) -> CSI2PoseModel:
    """Construct CSI2Pose from config values."""

    model_config = config["model"]
    return CSI2PoseModel(
        feature_dim=int(model_config["feature_dim"]),
        temporal_layers=int(model_config["temporal_layers"]),
        temporal_kernel_size=int(model_config["temporal_kernel_size"]),
        dropout=float(model_config["dropout"]),
        heatmap_size=int(model_config["heatmap_size"]),
        softargmax_temperature=float(model_config["softargmax_temperature"]),
    )


def build_target_heatmaps(
    keypoints: torch.Tensor,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    """Create Gaussian GT heatmaps from normalized keypoints.

    Args:
        keypoints: Normalized coordinates shaped ``B x T x 17 x 2``.
    """

    keypoints = keypoints.clamp(0.0, 1.0)
    coordinates = torch.arange(heatmap_size, dtype=keypoints.dtype, device=keypoints.device)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    centers_x = keypoints[..., 0] * (heatmap_size - 1)
    centers_y = keypoints[..., 1] * (heatmap_size - 1)
    distance_sq = (
        (grid_x.view(1, 1, 1, heatmap_size, heatmap_size) - centers_x[..., None, None]) ** 2
        + (grid_y.view(1, 1, 1, heatmap_size, heatmap_size) - centers_y[..., None, None]) ** 2
    )
    return torch.exp(-distance_sq / (2.0 * sigma * sigma))


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move the tensors needed by the model and losses to the target device."""

    return {
        "csi_amplitude": batch["csi_amplitude"].to(device=device, dtype=torch.float32),
        "csi_phase_cos": batch["csi_phase_cos"].to(device=device, dtype=torch.float32),
        "keypoints": batch["keypoints"].to(device=device, dtype=torch.float32),
    }


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine coordinate and heatmap supervision."""

    loss_config = config["loss"]
    target_heatmaps = build_target_heatmaps(
        targets,
        heatmap_size=int(config["model"]["heatmap_size"]),
        sigma=float(loss_config["heatmap_sigma"]),
    )
    coordinate_loss = nn.functional.smooth_l1_loss(outputs["keypoints"], targets)
    heatmap_loss = nn.functional.mse_loss(torch.sigmoid(outputs["heatmaps"]), target_heatmaps)
    total_loss = (
        float(loss_config["coordinate_weight"]) * coordinate_loss
        + float(loss_config["heatmap_weight"]) * heatmap_loss
    )
    return total_loss, {
        "loss": float(total_loss.detach().item()),
        "coordinate_loss": float(coordinate_loss.detach().item()),
        "heatmap_loss": float(heatmap_loss.detach().item()),
    }


def train_one_epoch(
    model: CSI2PoseModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    """Run one training epoch with tqdm progress feedback."""

    model.train()
    use_amp = bool(config["train"]["amp"]) and device.type == "cuda"
    progress = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True)
    totals = {"loss": 0.0, "coordinate_loss": 0.0, "heatmap_loss": 0.0}

    for step, raw_batch in enumerate(progress, start=1):
        batch = move_batch_to_device(raw_batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(batch["csi_amplitude"], batch["csi_phase_cos"])
            loss, loss_items = compute_loss(outputs, batch["keypoints"], config=config)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["train"]["grad_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()

        for name in totals:
            totals[name] += loss_items[name]
        progress.set_postfix({name: totals[name] / step for name in totals})

    return {name: value / max(len(loader), 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: CSI2PoseModel,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate loss and PCK on one split."""

    model.eval()
    totals = {"loss": 0.0, "coordinate_loss": 0.0, "heatmap_loss": 0.0}
    accumulator = PCKAccumulator(
        x_scale=loader.dataset.keypoint_x_scale,
        y_scale=loader.dataset.keypoint_y_scale,
        thresholds=config["metrics"]["pck_thresholds"],
    )
    progress = tqdm(loader, desc="evaluate", dynamic_ncols=True)

    for step, raw_batch in enumerate(progress, start=1):
        batch = move_batch_to_device(raw_batch, device=device)
        outputs = model(batch["csi_amplitude"], batch["csi_phase_cos"])
        _, loss_items = compute_loss(outputs, batch["keypoints"], config=config)
        accumulator.update(outputs["keypoints"], batch["keypoints"])
        for name in totals:
            totals[name] += loss_items[name]
        progress.set_postfix({name: totals[name] / step for name in totals})

    metrics = {name: value / max(len(loader), 1) for name, value in totals.items()}
    metrics.update(accumulator.compute())
    return metrics


def save_checkpoint(
    path: Path,
    model: CSI2PoseModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
) -> None:
    """Persist the model state needed for later testing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_pck@20": best_score,
            "config": config,
        },
        path,
    )


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Write final train/val/test metrics as JSON on the server run directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    run_dir = Path(args.run_dir or config["run_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(config["train"]["amp"]) and device.type == "cuda")

    train_loader = make_sequence_loader(args.dataset_root, "train", config=config, shuffle=True)
    val_loader = make_sequence_loader(args.dataset_root, "val", config=config, shuffle=False)
    test_loader = make_sequence_loader(args.dataset_root, "test", config=config, shuffle=False)

    best_score = -1.0
    best_checkpoint = run_dir / "best.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, config, epoch=epoch)
        val_metrics = evaluate(model, val_loader, device, config)
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        val_score = float(val_metrics["pck@20"])
        if val_score > best_score:
            best_score = val_score
            save_checkpoint(best_checkpoint, model, optimizer, epoch, best_score, config=config)

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, config)
    final_metrics = {"best_val_pck@20": best_score, "history": history, "test": test_metrics}
    write_metrics(run_dir / "metrics.json", final_metrics)
    print(json.dumps(final_metrics["test"], indent=2))
    return final_metrics


if __name__ == "__main__":
    main()
