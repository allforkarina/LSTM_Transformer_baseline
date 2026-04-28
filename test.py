from __future__ import annotations

"""Evaluate a trained CSI2Pose decoder checkpoint on one HDF5 split."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from train import (
    DECODER_NAMES,
    DEFAULT_DECODER,
    activate_decoder_config,
    build_model,
    build_skeleton_prior,
    evaluate,
    load_config,
    load_model_state,
    make_sequence_loader,
    skeleton_prior_from_state,
    skeleton_prior_to_device,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test a CSI2Pose decoder baseline")
    parser.add_argument("--config", type=str, default="configs/csi2pose_tcn.yaml", help="YAML config path")
    parser.add_argument("--dataset-root", type=str, required=True, help="Packed MM-Fi HDF5 dataset path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint created by train.py")
    parser.add_argument("--decoder", type=str, default=DEFAULT_DECODER, choices=DECODER_NAMES, help="Decoder to test")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"), help="Split to evaluate")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = activate_decoder_config(load_config(args.config), decoder=args.decoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    load_model_state(model, checkpoint["model_state_dict"], decoder=args.decoder)
    loader = make_sequence_loader(args.dataset_root, args.split, config=config, shuffle=False)
    if "skeleton_prior" in checkpoint:
        skeleton_prior = skeleton_prior_from_state(checkpoint["skeleton_prior"], device=device)
    else:
        train_loader = make_sequence_loader(args.dataset_root, "train", config=config, shuffle=False)
        skeleton_prior = skeleton_prior_to_device(build_skeleton_prior(train_loader.dataset), device=device)
    metrics = evaluate(model, loader, device, config, skeleton_prior=skeleton_prior)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
