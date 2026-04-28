from __future__ import annotations

"""Train CSI2Pose decoder baselines on windowed MM-Fi HDF5 data."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataloader import DEFAULT_SPLIT_SCHEME, MMFiPoseSequenceDataset
from metrics import PCKAccumulator
from models import CSI2PoseHeatmapModel, CSI2PoseRegressionModel


DECODER_NAMES = ("heatmap", "regression")
DEFAULT_DECODER = "regression"
SKELETON_EPS = 1.0e-6
BODY_BONES = (
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
TORSO_SCALE_BONES = ((5, 11), (6, 12))
BODY_ANGLE_TRIPLETS = (
    (5, 7, 9),
    (6, 8, 10),
    (11, 13, 15),
    (12, 14, 16),
    (11, 5, 7),
    (12, 6, 8),
    (5, 11, 13),
    (6, 12, 14),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CSI2Pose decoder baseline")
    parser.add_argument("--config", type=str, default="configs/csi2pose_tcn.yaml", help="YAML config path")
    parser.add_argument("--dataset-root", type=str, required=True, help="Packed MM-Fi HDF5 dataset path")
    parser.add_argument("--decoder", type=str, default=DEFAULT_DECODER, choices=DECODER_NAMES, help="Decoder to train")
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


def activate_decoder_config(config: dict[str, Any], decoder: str) -> dict[str, Any]:
    """Merge shared config with the selected decoder-specific config."""

    if decoder not in DECODER_NAMES:
        raise ValueError(f"decoder must be one of {DECODER_NAMES}, got {decoder}")

    decoder_configs = config.get("decoders")
    if not isinstance(decoder_configs, dict) or decoder not in decoder_configs:
        raise KeyError(f"Config must define decoders.{decoder}")

    decoder_config = decoder_configs[decoder]
    active_config = dict(config)
    active_config["decoder"] = decoder
    active_config["model"] = {
        **config.get("model", {}),
        **decoder_config.get("model", {}),
    }
    active_config["train"] = {
        **config.get("train", {}),
        **decoder_config.get("train", {}),
    }
    active_config["loss"] = dict(decoder_config["loss"])
    active_config["run_dir"] = decoder_config.get("run_dir", config.get("run_dir"))
    return active_config


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
    print(f"Building {split} windows...", flush=True)
    dataset = MMFiPoseSequenceDataset(
        dataset_root=dataset_root,
        split=split,
        split_scheme=data_config.get("split_scheme", DEFAULT_SPLIT_SCHEME),
        window_size=int(data_config["window_size"]),
        window_stride=int(data_config["window_stride"]),
    )
    print(f"{split} windows: {len(dataset)}", flush=True)
    return DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_config["num_workers"]),
        pin_memory=bool(train_config["pin_memory"]),
    )


def build_model(config: dict[str, Any]):
    """Construct CSI2Pose from config values."""

    model_config = config["model"]
    common_kwargs = {
        "feature_dim": int(model_config["feature_dim"]),
        "temporal_layers": int(model_config["temporal_layers"]),
        "temporal_kernel_size": int(model_config["temporal_kernel_size"]),
        "dropout": float(model_config["dropout"]),
    }
    if config["decoder"] == "heatmap":
        return CSI2PoseHeatmapModel(
            **common_kwargs,
            heatmap_size=int(model_config["heatmap_size"]),
            softargmax_temperature=float(model_config["softargmax_temperature"]),
        )
    if config["decoder"] == "regression":
        return CSI2PoseRegressionModel(**common_kwargs)
    raise ValueError(f"Unsupported decoder: {config['decoder']}")


def load_model_state(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], decoder: str) -> None:
    """Load current checkpoints and legacy heatmap checkpoints."""

    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        if decoder != "heatmap":
            raise

    remapped_state = {}
    for name, value in state_dict.items():
        if name.startswith(("frame_encoder.", "temporal_input.", "temporal_encoder.")):
            remapped_state[f"backbone.{name}"] = value
        else:
            remapped_state[name] = value
    model.load_state_dict(remapped_state)


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


def _pixel_keypoints(keypoints: torch.Tensor, x_scale: float, y_scale: float) -> torch.Tensor:
    """Restore normalized keypoints to pixel coordinates for skeleton geometry."""

    scale = torch.tensor([x_scale, y_scale], dtype=keypoints.dtype, device=keypoints.device)
    return keypoints * scale


def _bone_ratios(pixel_keypoints: torch.Tensor) -> torch.Tensor:
    """Return torso-scale-normalized body-bone lengths."""

    bone_lengths = []
    for start, end in BODY_BONES:
        bone_lengths.append(torch.linalg.vector_norm(pixel_keypoints[..., end, :] - pixel_keypoints[..., start, :], dim=-1))
    lengths = torch.stack(bone_lengths, dim=-1)

    torso_lengths = []
    for start, end in TORSO_SCALE_BONES:
        torso_lengths.append(torch.linalg.vector_norm(pixel_keypoints[..., end, :] - pixel_keypoints[..., start, :], dim=-1))
    pose_scale = torch.stack(torso_lengths, dim=-1).mean(dim=-1, keepdim=True).clamp_min(SKELETON_EPS)
    return lengths / pose_scale


def _angle_cosines(pixel_keypoints: torch.Tensor) -> torch.Tensor:
    """Return body-joint angle cosines for selected COCO17 triples."""

    cosines = []
    for first, center, last in BODY_ANGLE_TRIPLETS:
        first_vector = pixel_keypoints[..., first, :] - pixel_keypoints[..., center, :]
        last_vector = pixel_keypoints[..., last, :] - pixel_keypoints[..., center, :]
        numerator = torch.sum(first_vector * last_vector, dim=-1)
        denominator = (
            torch.linalg.vector_norm(first_vector, dim=-1)
            * torch.linalg.vector_norm(last_vector, dim=-1)
        ).clamp_min(SKELETON_EPS)
        cosines.append((numerator / denominator).clamp(-1.0, 1.0))
    return torch.stack(cosines, dim=-1)


def _summarize_skeleton_prior(pixel_keypoints: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute train-split skeleton prior statistics."""

    bone_values = _bone_ratios(pixel_keypoints).reshape(-1, len(BODY_BONES))
    angle_values = _angle_cosines(pixel_keypoints).reshape(-1, len(BODY_ANGLE_TRIPLETS))
    return {
        "bone_mean": bone_values.mean(dim=0),
        "bone_std": bone_values.std(dim=0, unbiased=False).clamp_min(SKELETON_EPS),
        "angle_mean": angle_values.mean(dim=0),
        "angle_std": angle_values.std(dim=0, unbiased=False).clamp_min(SKELETON_EPS),
    }


def build_skeleton_prior(dataset: MMFiPoseSequenceDataset) -> dict[str, torch.Tensor | float]:
    """Build train-split skeleton prior statistics from normalized HDF5 keypoints."""

    frame_dataset = dataset.frame_dataset
    keypoints = []
    for index in range(len(frame_dataset)):
        keypoints.append(torch.as_tensor(frame_dataset[index]["keypoints"], dtype=torch.float32))
    normalized_keypoints = torch.stack(keypoints, dim=0)
    pixel_keypoints = _pixel_keypoints(
        normalized_keypoints,
        x_scale=frame_dataset.keypoint_x_scale,
        y_scale=frame_dataset.keypoint_y_scale,
    )
    prior = _summarize_skeleton_prior(pixel_keypoints)
    prior["x_scale"] = float(frame_dataset.keypoint_x_scale)
    prior["y_scale"] = float(frame_dataset.keypoint_y_scale)
    return prior


def skeleton_prior_to_device(
    skeleton_prior: dict[str, torch.Tensor | float],
    device: torch.device,
) -> dict[str, torch.Tensor | float]:
    """Move tensor prior statistics to the training device."""

    return {
        name: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for name, value in skeleton_prior.items()
    }


def skeleton_prior_to_jsonable(skeleton_prior: dict[str, torch.Tensor | float]) -> dict[str, Any]:
    """Convert skeleton prior tensors to JSON-serializable values."""

    result: dict[str, Any] = {}
    for name, value in skeleton_prior.items():
        if isinstance(value, torch.Tensor):
            result[name] = value.detach().cpu().tolist()
        else:
            result[name] = value
    return result


def skeleton_prior_from_state(state: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor | float]:
    """Restore skeleton prior statistics saved in a checkpoint."""

    restored: dict[str, torch.Tensor | float] = {}
    for name, value in state.items():
        if name in {"x_scale", "y_scale"}:
            restored[name] = float(value)
        else:
            restored[name] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return restored


def compute_skeleton_prior_losses(
    predictions: torch.Tensor,
    skeleton_prior: dict[str, torch.Tensor | float],
) -> dict[str, torch.Tensor]:
    """Penalize implausible bone ratios and angle cosines under train-set priors."""

    pixel_predictions = _pixel_keypoints(
        predictions,
        x_scale=float(skeleton_prior["x_scale"]),
        y_scale=float(skeleton_prior["y_scale"]),
    )
    bone_z_scores = (
        _bone_ratios(pixel_predictions) - skeleton_prior["bone_mean"]
    ) / skeleton_prior["bone_std"]
    angle_z_scores = (
        _angle_cosines(pixel_predictions) - skeleton_prior["angle_mean"]
    ) / skeleton_prior["angle_std"]
    return {
        "bone_prior_loss": nn.functional.smooth_l1_loss(bone_z_scores, torch.zeros_like(bone_z_scores)),
        "angle_prior_loss": nn.functional.smooth_l1_loss(angle_z_scores, torch.zeros_like(angle_z_scores)),
    }


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
    skeleton_prior: dict[str, torch.Tensor | float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine decoder-specific supervision with skeleton priors."""

    loss_config = config["loss"]
    coordinate_loss = nn.functional.smooth_l1_loss(outputs["keypoints"], targets)
    prior_losses = compute_skeleton_prior_losses(outputs["keypoints"], skeleton_prior)
    total_loss = float(loss_config["coordinate_weight"]) * coordinate_loss
    total_loss = total_loss + float(loss_config["bone_weight"]) * prior_losses["bone_prior_loss"]
    total_loss = total_loss + float(loss_config["angle_weight"]) * prior_losses["angle_prior_loss"]
    loss_items = {
        "loss": float(total_loss.detach().item()),
        "coordinate_loss": float(coordinate_loss.detach().item()),
        "bone_prior_loss": float(prior_losses["bone_prior_loss"].detach().item()),
        "angle_prior_loss": float(prior_losses["angle_prior_loss"].detach().item()),
    }
    if config["decoder"] == "heatmap":
        target_heatmaps = build_target_heatmaps(
            targets,
            heatmap_size=int(config["model"]["heatmap_size"]),
            sigma=float(loss_config["heatmap_sigma"]),
        )
        heatmap_loss = nn.functional.mse_loss(torch.sigmoid(outputs["heatmaps"]), target_heatmaps)
        total_loss = total_loss + float(loss_config["heatmap_weight"]) * heatmap_loss
        loss_items["loss"] = float(total_loss.detach().item())
        loss_items["heatmap_loss"] = float(heatmap_loss.detach().item())
    return total_loss, loss_items


def format_epoch_summary(
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    best_score: float,
    saved_best: bool,
) -> str:
    """Return the compact per-epoch training summary shown in server logs."""

    suffix = " saved_best=True" if saved_best else ""
    return (
        f"epoch={epoch} "
        f"train_loss={float(train_metrics['loss']):.6f} "
        f"val_loss={float(val_metrics['loss']):.6f} "
        f"val_pck@20={float(val_metrics['pck@20']):.6f} "
        f"best_val_pck@20={best_score:.6f}"
        f"{suffix}"
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: dict[str, Any],
    skeleton_prior: dict[str, torch.Tensor | float],
    epoch: int,
) -> dict[str, float]:
    """Run one training epoch with tqdm progress feedback."""

    model.train()
    use_amp = bool(config["train"]["amp"]) and device.type == "cuda"
    progress = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True)
    totals: dict[str, float] = {}

    for step, raw_batch in enumerate(progress, start=1):
        batch = move_batch_to_device(raw_batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch["csi_amplitude"], batch["csi_phase_cos"])
            loss, loss_items = compute_loss(
                outputs,
                batch["keypoints"],
                config=config,
                skeleton_prior=skeleton_prior,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["train"]["grad_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()

        for name, value in loss_items.items():
            totals[name] = totals.get(name, 0.0) + value
        progress.set_postfix({"loss": totals["loss"] / step})

    return {name: value / max(len(loader), 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    skeleton_prior: dict[str, torch.Tensor | float],
) -> dict[str, Any]:
    """Evaluate loss and PCK on one split."""

    model.eval()
    totals: dict[str, float] = {}
    accumulator = PCKAccumulator(
        x_scale=loader.dataset.keypoint_x_scale,
        y_scale=loader.dataset.keypoint_y_scale,
        thresholds=config["metrics"]["pck_thresholds"],
    )
    progress = tqdm(loader, desc="evaluate", dynamic_ncols=True)

    for step, raw_batch in enumerate(progress, start=1):
        batch = move_batch_to_device(raw_batch, device=device)
        outputs = model(batch["csi_amplitude"], batch["csi_phase_cos"])
        _, loss_items = compute_loss(
            outputs,
            batch["keypoints"],
            config=config,
            skeleton_prior=skeleton_prior,
        )
        accumulator.update(outputs["keypoints"], batch["keypoints"])
        for name, value in loss_items.items():
            totals[name] = totals.get(name, 0.0) + value
        progress.set_postfix({"loss": totals["loss"] / step})

    metrics = {name: value / max(len(loader), 1) for name, value in totals.items()}
    metrics.update(accumulator.compute())
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
    skeleton_prior: dict[str, torch.Tensor | float],
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
            "skeleton_prior": skeleton_prior_to_jsonable(skeleton_prior),
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
    config = activate_decoder_config(load_config(args.config), decoder=args.decoder)
    set_seed(int(config["seed"]))

    run_dir = Path(args.run_dir or config["run_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = GradScaler("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")

    train_loader = make_sequence_loader(args.dataset_root, "train", config=config, shuffle=True)
    val_loader = make_sequence_loader(args.dataset_root, "val", config=config, shuffle=False)
    skeleton_prior = skeleton_prior_to_device(build_skeleton_prior(train_loader.dataset), device=device)

    best_score = -1.0
    best_checkpoint = run_dir / "best.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            skeleton_prior=skeleton_prior,
            epoch=epoch,
        )
        val_metrics = evaluate(model, val_loader, device, config, skeleton_prior=skeleton_prior)
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        val_score = float(val_metrics["pck@20"])
        saved_best = False
        if val_score > best_score:
            best_score = val_score
            saved_best = True
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                best_score,
                config=config,
                skeleton_prior=skeleton_prior,
            )
        print(
            format_epoch_summary(
                epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_score=best_score,
                saved_best=saved_best,
            ),
            flush=True,
        )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    load_model_state(model, checkpoint["model_state_dict"], decoder=args.decoder)
    test_loader = make_sequence_loader(args.dataset_root, "test", config=config, shuffle=False)
    test_metrics = evaluate(model, test_loader, device, config, skeleton_prior=skeleton_prior)
    final_metrics = {
        "decoder": args.decoder,
        "best_val_pck@20": best_score,
        "skeleton_prior": skeleton_prior_to_jsonable(skeleton_prior),
        "history": history,
        "test": test_metrics,
    }
    write_metrics(run_dir / "metrics.json", final_metrics)
    print(json.dumps(final_metrics["test"], indent=2))
    return final_metrics


if __name__ == "__main__":
    main()
