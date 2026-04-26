from __future__ import annotations

import importlib

import numpy as np

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
    assert not hasattr(args, "subset_size")


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
    assert args.batch_size == 128
    assert not hasattr(args, "num_visualizations")


def test_build_csi_heatmap_stacks_antennas() -> None:
    eval_module = importlib.import_module("eval")
    csi_amplitude = np.arange(3 * 114 * 10, dtype=np.float32).reshape(3, 114, 10)

    heatmap = eval_module._build_csi_heatmap(csi_amplitude)

    assert heatmap.shape == (342, 10)
    assert np.array_equal(heatmap[:114], csi_amplitude[0])
    assert np.array_equal(heatmap[114:228], csi_amplitude[1])
    assert np.array_equal(heatmap[228:], csi_amplitude[2])
