from __future__ import annotations

import h5py
import numpy as np
import torch

import test as test_script
import train as train_script
from metrics import compute_pck
from models import CSI2PoseHeatmapModel, CSI2PoseRegressionModel
from train import (
    _summarize_skeleton_prior,
    activate_decoder_config,
    build_target_heatmaps,
    compute_loss,
)
from visualization import save_pose_comparison, select_one_frame_per_action_env_sample


def _base_config() -> dict:
    return {
        "seed": 42,
        "data": {"split_scheme": "action_env", "window_size": 2, "window_stride": 1},
        "model": {"feature_dim": 32, "temporal_layers": 1, "temporal_kernel_size": 3, "dropout": 0.0},
        "metrics": {"pck_thresholds": [5, 10]},
        "train": {"epochs": 1, "batch_size": 2},
        "decoders": {
            "heatmap": {
                "run_dir": "runs/heatmap",
                "model": {"heatmap_size": 16, "softargmax_temperature": 0.05},
                "loss": {
                    "coordinate_weight": 1.0,
                    "heatmap_weight": 0.1,
                    "heatmap_sigma": 1.5,
                    "bone_weight": 0.05,
                    "angle_weight": 0.02,
                },
            },
            "regression": {
                "run_dir": "runs/regression",
                "loss": {"coordinate_weight": 1.0, "bone_weight": 0.05, "angle_weight": 0.02},
            },
        },
    }


def _sample_csi() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(2, 3, 3, 114, 10), torch.rand(2, 3, 3, 114, 10)


def _sample_keypoints() -> torch.Tensor:
    keypoints = torch.zeros(2, 3, 17, 2)
    keypoints[..., :, 0] = torch.linspace(0.2, 0.8, 17)
    keypoints[..., :, 1] = torch.linspace(0.1, 0.9, 17)
    return keypoints


def _sample_skeleton_prior() -> dict[str, torch.Tensor | float]:
    keypoints = _sample_keypoints().reshape(-1, 17, 2) * 100.0
    prior = _summarize_skeleton_prior(keypoints)
    prior["x_scale"] = 100.0
    prior["y_scale"] = 100.0
    return prior


def test_heatmap_model_forward_shapes() -> None:
    model = CSI2PoseHeatmapModel(feature_dim=32, temporal_layers=1, heatmap_size=16, dropout=0.0)
    model.eval()
    csi_amplitude, csi_phase_cos = _sample_csi()

    with torch.no_grad():
        outputs = model(csi_amplitude, csi_phase_cos)

    assert outputs["keypoints"].shape == (2, 3, 17, 2)
    assert outputs["heatmaps"].shape == (2, 3, 17, 16, 16)
    assert torch.all(outputs["keypoints"] >= 0.0)
    assert torch.all(outputs["keypoints"] <= 1.0)


def test_regression_model_forward_shapes() -> None:
    model = CSI2PoseRegressionModel(feature_dim=32, temporal_layers=1, dropout=0.0)
    model.eval()
    csi_amplitude, csi_phase_cos = _sample_csi()

    with torch.no_grad():
        outputs = model(csi_amplitude, csi_phase_cos)

    assert outputs["keypoints"].shape == (2, 3, 17, 2)
    assert "heatmaps" not in outputs


def test_target_heatmaps_follow_keypoint_shape() -> None:
    keypoints = torch.full((2, 3, 17, 2), 0.5)
    heatmaps = build_target_heatmaps(keypoints, heatmap_size=16, sigma=1.5)

    assert heatmaps.shape == (2, 3, 17, 16, 16)
    assert torch.all(heatmaps >= 0.0)
    assert torch.all(heatmaps <= 1.0)


def test_heatmap_loss_reports_decoder_specific_items() -> None:
    config = activate_decoder_config(_base_config(), "heatmap")
    outputs = {
        "keypoints": _sample_keypoints(),
        "heatmaps": torch.zeros(2, 3, 17, 16, 16),
    }
    _, loss_items = compute_loss(
        outputs,
        _sample_keypoints(),
        config=config,
        skeleton_prior=_sample_skeleton_prior(),
    )

    assert set(loss_items) == {"loss", "coordinate_loss", "bone_prior_loss", "angle_prior_loss", "heatmap_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in loss_items.values())


def test_regression_loss_reports_decoder_specific_items() -> None:
    config = activate_decoder_config(_base_config(), "regression")
    outputs = {"keypoints": _sample_keypoints()}
    _, loss_items = compute_loss(
        outputs,
        _sample_keypoints(),
        config=config,
        skeleton_prior=_sample_skeleton_prior(),
    )

    assert set(loss_items) == {"loss", "coordinate_loss", "bone_prior_loss", "angle_prior_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in loss_items.values())


def test_skeleton_prior_handles_degenerate_keypoints() -> None:
    keypoints = torch.zeros(2, 17, 2)
    prior = _summarize_skeleton_prior(keypoints)

    for value in prior.values():
        assert torch.all(torch.isfinite(value))


def test_pck_reports_overall_and_per_joint_scores() -> None:
    targets = torch.zeros(1, 1, 17, 2)
    predictions = targets.clone()
    predictions[..., 0] = 0.05

    metrics = compute_pck(predictions, targets, x_scale=100.0, y_scale=100.0, thresholds=(5.0, 10.0))

    assert metrics["pck@5"] == 1.0
    assert metrics["pck@10"] == 1.0
    assert metrics["per_joint"]["nose"]["pck@5"] == 1.0


def test_train_parser_accepts_decoder_args() -> None:
    args = train_script.parse_args(["--dataset-root", "data/mmfi_pose.h5"])
    heatmap_args = train_script.parse_args(["--dataset-root", "data/mmfi_pose.h5", "--decoder", "heatmap"])

    assert args.decoder == "regression"
    assert heatmap_args.decoder == "heatmap"
    assert args.config == "configs/csi2pose_tcn.yaml"
    assert args.dataset_root == "data/mmfi_pose.h5"


def test_test_parser_accepts_decoder_args() -> None:
    args = test_script.parse_args(
        [
            "--dataset-root",
            "data/mmfi_pose.h5",
            "--checkpoint",
            "runs/csi2pose_regression_tcn/best.pt",
            "--visualize",
            "--visualization-dir",
            "runs/visualizations",
            "--visualization-seed",
            "7",
        ]
    )

    assert args.decoder == "regression"
    assert args.split == "test"
    assert args.checkpoint == "runs/csi2pose_regression_tcn/best.pt"
    assert args.visualize is True
    assert args.visualization_dir == "runs/visualizations"
    assert args.visualization_seed == 7


def test_visualization_selection_samples_one_middle_frame_per_group(tmp_path) -> None:
    dataset_path = tmp_path / "mini.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(dataset_path, "w") as h5_file:
        h5_file.create_dataset("action", data=np.asarray(["A01", "A01", "A01", "A01"], dtype=object), dtype=string_dtype)
        h5_file.create_dataset("sample", data=np.asarray(["S01", "S01", "S02", "S02"], dtype=object), dtype=string_dtype)
        h5_file.create_dataset(
            "environment",
            data=np.asarray(["env1", "env1", "env1", "env1"], dtype=object),
            dtype=string_dtype,
        )
        h5_file.create_dataset(
            "frame_id",
            data=np.asarray(["frame001", "frame002", "frame001", "frame002"], dtype=object),
            dtype=string_dtype,
        )

    class FakeDataset:
        dataset_root = dataset_path
        windows = [(0, 1), (2, 3)]

    selected = select_one_frame_per_action_env_sample(FakeDataset(), seed=42)

    assert len(selected) == 2
    assert {(frame.action, frame.environment, frame.sample) for frame in selected} == {
        ("A01", "env1", "S01"),
        ("A01", "env1", "S02"),
    }
    assert {frame.frame_id for frame in selected} == {"frame002"}


def test_pose_comparison_visualization_denormalizes_and_saves(tmp_path) -> None:
    keypoints = torch.zeros(17, 2)
    keypoints[:, 0] = torch.linspace(0.1, 0.9, 17)
    keypoints[:, 1] = torch.linspace(0.2, 0.8, 17)
    output_path = tmp_path / "pose.png"

    save_pose_comparison(
        gt_keypoints=keypoints,
        predicted_keypoints=keypoints + 0.01,
        x_scale=100.0,
        y_scale=200.0,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
