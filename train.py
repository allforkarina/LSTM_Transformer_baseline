from __future__ import annotations

"""Training entrypoint for the frame-level MM-Fi LSTM + Transformer baseline."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from baseline_common import (
    build_dataloader,
    build_model,
    ensure_output_dir,
    get_dataset_scales,
    maybe_subset_dataset,
    model_config_to_dict,
    resolve_device,
    run_epoch,
    set_random_seed,
)
from dataloader import MMFiPoseDataset, SPLIT_SCHEMES


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for baseline training."""

    parser = argparse.ArgumentParser(description="Train the MM-Fi LSTM + Transformer baseline")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to one HDF5 dataset file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for logs and checkpoints")
    parser.add_argument(
        "--split-scheme",
        type=str,
        default="action_env",
        choices=SPLIT_SCHEMES,
        help="Dataset split scheme to train and validate on",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of PyTorch dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optional deterministic subset size for quick sanity or overfit checks",
    )
    parser.add_argument("--device", type=str, default="auto", help="Runtime device, for example auto, cuda, or cpu")
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early stopping patience measured in validation epochs without improvement",
    )
    parser.add_argument(
        "--smooth-l1-beta",
        type=float,
        default=0.02,
        help="Beta parameter for SmoothL1Loss",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training."""

    return build_arg_parser().parse_args(argv)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mpjpe: float,
    config: dict,
) -> None:
    """Persist one training checkpoint."""

    torch.save(
        {
            "epoch": epoch,
            "best_val_mpjpe": best_val_mpjpe,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )


def main(argv: list[str] | None = None) -> dict[str, float]:
    """Train the baseline model and write logs plus checkpoints."""

    args = parse_args(argv)
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_output_dir(args.output_dir)

    train_dataset = MMFiPoseDataset(dataset_root=args.dataset_root, split="train", split_scheme=args.split_scheme)
    val_dataset = MMFiPoseDataset(dataset_root=args.dataset_root, split="val", split_scheme=args.split_scheme)
    test_dataset = MMFiPoseDataset(dataset_root=args.dataset_root, split="test", split_scheme=args.split_scheme)

    train_dataset_view = maybe_subset_dataset(train_dataset, subset_size=args.subset_size, seed=args.seed)
    val_dataset_view = maybe_subset_dataset(val_dataset, subset_size=args.subset_size, seed=args.seed + 1)
    test_dataset_view = maybe_subset_dataset(test_dataset, subset_size=args.subset_size, seed=args.seed + 2)

    train_loader = build_dataloader(train_dataset_view, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_dataloader(val_dataset_view, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_dataloader(test_dataset_view, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    x_scale, y_scale = get_dataset_scales(train_dataset_view)
    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.SmoothL1Loss(beta=args.smooth_l1_beta)

    model_config = model_config_to_dict(model.config)
    run_config = {
        "dataset_root": args.dataset_root,
        "output_dir": str(output_dir),
        "split_scheme": args.split_scheme,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "subset_size": args.subset_size,
        "device": str(device),
        "patience": args.patience,
        "smooth_l1_beta": args.smooth_l1_beta,
        "model": model_config,
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    log_path = output_dir / "train_log.csv"
    best_checkpoint_path = output_dir / "best_val_mpjpe.pth"
    last_checkpoint_path = output_dir / "last.pth"
    best_val_mpjpe = float("inf")
    epochs_without_improvement = 0
    best_metrics: dict[str, float] = {}

    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_mpjpe",
                "train_pck_0.05",
                "train_pck_0.10",
                "train_pck_0.20",
                "val_loss",
                "val_mpjpe",
                "val_pck_0.05",
                "val_pck_0.10",
                "val_pck_0.20",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device=device,
                criterion=criterion,
                x_scale=x_scale,
                y_scale=y_scale,
                optimizer=optimizer,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                criterion=criterion,
                x_scale=x_scale,
                y_scale=y_scale,
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_metrics["loss"],
                    "train_mpjpe": train_metrics["mpjpe"],
                    "train_pck_0.05": train_metrics["pck_0.05"],
                    "train_pck_0.10": train_metrics["pck_0.10"],
                    "train_pck_0.20": train_metrics["pck_0.20"],
                    "val_loss": val_metrics["loss"],
                    "val_mpjpe": val_metrics["mpjpe"],
                    "val_pck_0.05": val_metrics["pck_0.05"],
                    "val_pck_0.10": val_metrics["pck_0.10"],
                    "val_pck_0.20": val_metrics["pck_0.20"],
                }
            )
            csv_file.flush()

            save_checkpoint(
                checkpoint_path=last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mpjpe=best_val_mpjpe,
                config=run_config,
            )

            if val_metrics["mpjpe"] < best_val_mpjpe:
                best_val_mpjpe = val_metrics["mpjpe"]
                epochs_without_improvement = 0
                best_metrics = val_metrics
                save_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_mpjpe=best_val_mpjpe,
                    config=run_config,
                )
            else:
                epochs_without_improvement += 1

            scheduler.step()
            if epochs_without_improvement >= args.patience:
                break

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    metrics_path = output_dir / "best_metrics.json"
    metrics_payload = {
        "best_val_mpjpe": best_val_mpjpe,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return {
        "best_val_mpjpe": best_val_mpjpe,
        "test_mpjpe": test_metrics["mpjpe"],
    }


if __name__ == "__main__":
    main()
