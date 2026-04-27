from __future__ import annotations

import torch

import test as test_script
import train as train_script
from metrics import compute_pck
from models import CSI2PoseModel
from train import build_target_heatmaps


def test_csi2pose_forward_shapes() -> None:
    model = CSI2PoseModel(feature_dim=32, temporal_layers=1, heatmap_size=16, dropout=0.0)
    model.eval()
    csi_amplitude = torch.rand(2, 3, 3, 114, 10)
    csi_phase_cos = torch.rand(2, 3, 3, 114, 10)

    with torch.no_grad():
        outputs = model(csi_amplitude, csi_phase_cos)

    assert outputs["keypoints"].shape == (2, 3, 17, 2)
    assert outputs["heatmaps"].shape == (2, 3, 17, 16, 16)
    assert torch.all(outputs["keypoints"] >= 0.0)
    assert torch.all(outputs["keypoints"] <= 1.0)


def test_target_heatmaps_follow_keypoint_shape() -> None:
    keypoints = torch.full((2, 3, 17, 2), 0.5)
    heatmaps = build_target_heatmaps(keypoints, heatmap_size=16, sigma=1.5)

    assert heatmaps.shape == (2, 3, 17, 16, 16)
    assert torch.all(heatmaps >= 0.0)
    assert torch.all(heatmaps <= 1.0)


def test_pck_reports_overall_and_per_joint_scores() -> None:
    targets = torch.zeros(1, 1, 17, 2)
    predictions = targets.clone()
    predictions[..., 0] = 0.05

    metrics = compute_pck(predictions, targets, x_scale=100.0, y_scale=100.0, thresholds=(5.0, 10.0))

    assert metrics["pck@5"] == 1.0
    assert metrics["pck@10"] == 1.0
    assert metrics["per_joint"]["nose"]["pck@5"] == 1.0


def test_train_parser_accepts_required_args() -> None:
    args = train_script.parse_args(["--dataset-root", "data/mmfi_pose.h5"])

    assert args.config == "configs/csi2pose_tcn.yaml"
    assert args.dataset_root == "data/mmfi_pose.h5"


def test_test_parser_accepts_required_args() -> None:
    args = test_script.parse_args(
        [
            "--dataset-root",
            "data/mmfi_pose.h5",
            "--checkpoint",
            "runs/csi2pose_tcn/best.pt",
        ]
    )

    assert args.split == "test"
    assert args.checkpoint == "runs/csi2pose_tcn/best.pt"
