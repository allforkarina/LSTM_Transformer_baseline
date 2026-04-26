from __future__ import annotations

import importlib

import train


def test_train_parser_defaults() -> None:
    args = train.parse_args(
        [
            "--dataset-root",
            "data/mmfi_pose.h5",
            "--output-dir",
            "outputs/train",
        ]
    )

    assert args.split_scheme == "action_env"
    assert args.epochs == 60
    assert args.batch_size == 128


def test_eval_parser_defaults() -> None:
    eval_module = importlib.import_module("eval")
    args = eval_module.parse_args(
        [
            "--dataset-root",
            "data/mmfi_pose.h5",
            "--checkpoint",
            "outputs/train/best_val_mpjpe.pth",
            "--output-dir",
            "outputs/eval",
        ]
    )

    assert args.split == "test"
    assert args.num_visualizations == 8
    assert args.batch_size == 128
