from __future__ import annotations

"""Training entrypoint for the frame-level MM-Fi LSTM + Transformer baseline."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from baseline_common import (
    build_dataloader,
    build_model,
    ensure_output_dir,
    get_dataset_scales,
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


def build_datasets(dataset_root: str, split_scheme: str) -> tuple[MMFiPoseDataset, MMFiPoseDataset, MMFiPoseDataset]:
    """Create the train, validation, and test dataset objects for one split scheme."""

    train_dataset = MMFiPoseDataset(dataset_root=dataset_root, split="train", split_scheme=split_scheme)
    val_dataset = MMFiPoseDataset(dataset_root=dataset_root, split="val", split_scheme=split_scheme)
    test_dataset = MMFiPoseDataset(dataset_root=dataset_root, split="test", split_scheme=split_scheme)
    return train_dataset, val_dataset, test_dataset


def build_log_writer(csv_file) -> csv.DictWriter:
    """Create the CSV writer used for epoch-by-epoch training logs."""

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch",
            "lr",
            "elapsed_seconds",
            "train_loss",
            "train_mpjpe_px",
            "train_pck05",
            "train_pck10",
            "train_pck20",
            "val_loss",
            "val_mpjpe_px",
            "val_pck05",
            "val_pck10",
            "val_pck20",
            "best_val_mpjpe_px",
            "is_best",
            "epochs_without_improvement",
        ],
    )
    writer.writeheader()
    return writer


def format_log_row(
    *,
    epoch: int,
    lr: float,
    elapsed_seconds: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_mpjpe: float,
    is_best: bool,
    epochs_without_improvement: int,
) -> dict[str, str]:
    """Format one epoch log row with compact, human-readable numeric precision."""

    return {
        "epoch": str(epoch),
        "lr": f"{lr:.8f}",
        "elapsed_seconds": f"{elapsed_seconds:.2f}",
        "train_loss": f"{train_metrics['loss']:.6f}",
        "train_mpjpe_px": f"{train_metrics['mpjpe']:.4f}",
        "train_pck05": f"{train_metrics['pck_0.05']:.6f}",
        "train_pck10": f"{train_metrics['pck_0.10']:.6f}",
        "train_pck20": f"{train_metrics['pck_0.20']:.6f}",
        "val_loss": f"{val_metrics['loss']:.6f}",
        "val_mpjpe_px": f"{val_metrics['mpjpe']:.4f}",
        "val_pck05": f"{val_metrics['pck_0.05']:.6f}",
        "val_pck10": f"{val_metrics['pck_0.10']:.6f}",
        "val_pck20": f"{val_metrics['pck_0.20']:.6f}",
        "best_val_mpjpe_px": f"{best_val_mpjpe:.4f}",
        "is_best": "1" if is_best else "0",
        "epochs_without_improvement": str(epochs_without_improvement),
    }


def main(argv: list[str] | None = None) -> dict[str, float]:
    """Train the baseline model and write logs plus checkpoints."""

    args = parse_args(argv)
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_output_dir(args.output_dir)

    # The baseline always trains on the full split. This keeps the runtime contract
    # simple and avoids keeping a separate tiny-overfit workflow in the codebase.
    train_dataset, val_dataset, test_dataset = build_datasets(
        dataset_root=args.dataset_root,
        split_scheme=args.split_scheme,
    )

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # The HDF5 file stores normalization statistics derived from the train split.
    # They are reused here for both metrics and final prediction restoration.
    x_scale, y_scale = get_dataset_scales(train_dataset)
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
    training_start_time = time.perf_counter()

    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = build_log_writer(csv_file)

        # The outer progress bar shows high-level epoch progress. Each split uses its
        # own inner progress bar inside ``run_epoch`` so stalled jobs are easier to
        # diagnose on the Linux server.
        epoch_progress = tqdm(range(1, args.epochs + 1), desc="epochs", dynamic_ncols=True)
        for epoch in epoch_progress:
            train_metrics = run_epoch(
                model,
                train_loader,
                device=device,
                criterion=criterion,
                x_scale=x_scale,
                y_scale=y_scale,
                phase_name=f"train {epoch:03d}",
                optimizer=optimizer,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                criterion=criterion,
                x_scale=x_scale,
                y_scale=y_scale,
                phase_name=f"val   {epoch:03d}",
            )

            is_best = False
            if val_metrics["mpjpe"] < best_val_mpjpe:
                best_val_mpjpe = val_metrics["mpjpe"]
                epochs_without_improvement = 0
                best_metrics = val_metrics
                is_best = True
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

            save_checkpoint(
                checkpoint_path=last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mpjpe=best_val_mpjpe,
                config=run_config,
            )

            writer.writerow(
                format_log_row(
                    epoch=epoch,
                    lr=optimizer.param_groups[0]["lr"],
                    elapsed_seconds=time.perf_counter() - training_start_time,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_val_mpjpe=best_val_mpjpe,
                    is_best=is_best,
                    epochs_without_improvement=epochs_without_improvement,
                )
            )
            csv_file.flush()
            tqdm.write(
                f"[Epoch {epoch}/{args.epochs}] "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"train_pck@20={train_metrics['pck_0.20']:.6f} "
                f"val_pck@20={val_metrics['pck_0.20']:.6f}"
            )
            epoch_progress.set_postfix(
                train_mpjpe=f"{train_metrics['mpjpe']:.2f}",
                val_mpjpe=f"{val_metrics['mpjpe']:.2f}",
                best=f"{best_val_mpjpe:.2f}",
            )
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
        phase_name="test",
    )
    metrics_path = output_dir / "best_metrics.json"
    metrics_payload = {
        "best_val_mpjpe": best_val_mpjpe,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "stopped_epoch": epoch,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return {
        "best_val_mpjpe": best_val_mpjpe,
        "test_mpjpe": test_metrics["mpjpe"],
    }


if __name__ == "__main__":
    main()
