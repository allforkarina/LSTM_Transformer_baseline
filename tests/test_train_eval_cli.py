from __future__ import annotations

import importlib

import h5py
import numpy as np
import torch

from baseline_common import PoseSequenceLoss, compute_batch_statistics
from dataloader import MMFiPoseSequenceDataset
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
    assert args.batch_size == 64
    assert args.window_size == 16
    assert args.window_stride == 4
    assert args.bone_loss_weight == 0.1
    assert args.temporal_loss_weight == 0.05
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
    assert args.batch_size == 64
    assert args.window_size is None
    assert args.window_stride is None
    assert not hasattr(args, "num_visualizations")


def test_build_csi_heatmap_stacks_antennas() -> None:
    eval_module = importlib.import_module("eval")
    csi_amplitude = np.arange(3 * 114 * 10, dtype=np.float32).reshape(3, 114, 10)

    heatmap = eval_module._build_csi_heatmap(csi_amplitude)

    assert heatmap.shape == (342, 10)
    assert np.array_equal(heatmap[:114], csi_amplitude[0])
    assert np.array_equal(heatmap[114:228], csi_amplitude[1])
    assert np.array_equal(heatmap[228:], csi_amplitude[2])


def test_pose_sequence_loss_returns_finite_parts() -> None:
    criterion = PoseSequenceLoss(beta=0.02, bone_weight=0.1, temporal_weight=0.05)
    predictions = torch.rand(2, 4, 17, 2)
    targets = torch.rand(2, 4, 17, 2)

    loss_parts = criterion(predictions, targets)

    assert set(loss_parts) == {"loss", "pose_loss", "bone_loss", "temp_loss"}
    assert all(torch.isfinite(value) for value in loss_parts.values())


def test_pixel_pck_uses_pixel_thresholds() -> None:
    targets = torch.zeros(1, 1, 2, 2)
    predictions = torch.tensor([[[[0.0, 0.0], [0.1, 0.0]]]])

    stats = compute_batch_statistics(predictions, targets, x_scale=100.0, y_scale=100.0)

    assert stats["num_keypoints"] == 2.0
    assert stats["pck_10px_correct"] == 2.0
    assert stats["pck_20px_correct"] == 2.0
    assert stats["pck_30px_correct"] == 2.0


def test_sequence_dataset_builds_contiguous_windows(tmp_path) -> None:
    dataset_path = tmp_path / "synthetic.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(dataset_path, "w") as h5_file:
        h5_file.create_dataset("keypoints", data=np.zeros((6, 17, 2), dtype=np.float32))
        h5_file.create_dataset("csi_amplitude", data=np.zeros((6, 3, 114, 10), dtype=np.float32))
        h5_file.create_dataset("csi_phase", data=np.zeros((6, 3, 114, 10), dtype=np.float32))
        h5_file.create_dataset("csi_phase_cos", data=np.zeros((6, 3, 114, 10), dtype=np.float32))
        h5_file.create_dataset("action", data=np.array(["A01"] * 6, dtype=object), dtype=string_dtype)
        h5_file.create_dataset("sample", data=np.array(["S01"] * 3 + ["S02"] * 3, dtype=object), dtype=string_dtype)
        h5_file.create_dataset("environment", data=np.array(["env1"] * 6, dtype=object), dtype=string_dtype)
        h5_file.create_dataset(
            "frame_id",
            data=np.array(["frame001", "frame002", "frame003"] * 2, dtype=object),
            dtype=string_dtype,
        )
        h5_file.create_dataset("action_env_train_indices", data=np.arange(6, dtype=np.int64))
        h5_file.attrs["storage_format"] = "normalized"
        h5_file.attrs["keypoint_x_scale"] = 100.0
        h5_file.attrs["keypoint_y_scale"] = 200.0

    dataset = MMFiPoseSequenceDataset(dataset_path, split="train", window_size=2, window_stride=1)

    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["keypoints"].shape == (2, 17, 2)
    assert sample["csi_amplitude"].shape == (2, 3, 114, 10)
