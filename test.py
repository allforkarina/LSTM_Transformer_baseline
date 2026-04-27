from __future__ import annotations

"""Evaluate a trained CSI2Pose checkpoint on one HDF5 split."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from train import build_model, evaluate, load_config, make_sequence_loader


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test the CSI2Pose heatmap baseline")
    parser.add_argument("--config", type=str, default="configs/csi2pose_tcn.yaml", help="YAML config path")
    parser.add_argument("--dataset-root", type=str, required=True, help="Packed MM-Fi HDF5 dataset path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint created by train.py")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"), help="Split to evaluate")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_sequence_loader(args.dataset_root, args.split, config=config, shuffle=False)
    metrics = evaluate(model, loader, device, config)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
