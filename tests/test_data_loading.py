from __future__ import annotations

import h5py
import importlib.util
import numpy as np
from pathlib import Path

from dataloader import MMFiPoseDataset, MMFiPoseSequenceDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_H5_SCRIPT = PROJECT_ROOT / "scripts" / "build_h5_dataset.py"


def _load_build_h5_script():
    spec = importlib.util.spec_from_file_location("build_h5_dataset_script", BUILD_H5_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_h5(dataset_path, storage_format: str = "normalized") -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(dataset_path, "w") as h5_file:
        keypoints = np.zeros((6, 17, 2), dtype=np.float32)
        keypoints[..., 0] = 50.0
        keypoints[..., 1] = 100.0
        h5_file.create_dataset("keypoints", data=keypoints)
        h5_file.create_dataset("csi_amplitude", data=np.ones((6, 3, 114, 10), dtype=np.float32))
        h5_file.create_dataset("csi_phase", data=np.zeros((6, 3, 114, 10), dtype=np.float32))
        h5_file.create_dataset("csi_phase_cos", data=np.ones((6, 3, 114, 10), dtype=np.float32))
        h5_file.create_dataset("action", data=np.array(["A01"] * 6, dtype=object), dtype=string_dtype)
        h5_file.create_dataset("sample", data=np.array(["S01"] * 3 + ["S02"] * 3, dtype=object), dtype=string_dtype)
        h5_file.create_dataset("environment", data=np.array(["env1"] * 6, dtype=object), dtype=string_dtype)
        h5_file.create_dataset(
            "frame_id",
            data=np.array(["frame001", "frame002", "frame003"] * 2, dtype=object),
            dtype=string_dtype,
        )
        h5_file.create_dataset("action_env_train_indices", data=np.arange(6, dtype=np.int64))
        h5_file.create_dataset("action_env_val_indices", data=np.array([], dtype=np.int64))
        h5_file.create_dataset("action_env_test_indices", data=np.array([], dtype=np.int64))
        h5_file.attrs["storage_format"] = storage_format
        h5_file.attrs["amplitude_normalization"] = "train_split_min_max"
        h5_file.attrs["amplitude_train_min"] = 0.0
        h5_file.attrs["amplitude_train_max"] = 2.0
        h5_file.attrs["keypoint_x_scale"] = 100.0
        h5_file.attrs["keypoint_y_scale"] = 200.0


def test_frame_dataset_loads_hdf5_sample(tmp_path) -> None:
    dataset_path = tmp_path / "synthetic.h5"
    _write_synthetic_h5(dataset_path)

    dataset = MMFiPoseDataset(dataset_path, split="train")
    sample = dataset[0]

    assert sample["keypoints"].shape == (17, 2)
    assert sample["csi_amplitude"].shape == (3, 114, 10)
    assert sample["csi_phase_cos"].shape == (3, 114, 10)
    assert sample["action"] == "A01"
    assert sample["sample"] == "S01"
    assert sample["environment"] == "env1"


def test_raw_storage_is_normalized_on_load(tmp_path) -> None:
    dataset_path = tmp_path / "synthetic_raw.h5"
    _write_synthetic_h5(dataset_path, storage_format="cleaned_raw")

    dataset = MMFiPoseDataset(dataset_path, split="train")
    sample = dataset[0]

    assert np.allclose(sample["keypoints"][..., 0], 0.5)
    assert np.allclose(sample["keypoints"][..., 1], 0.5)
    assert np.allclose(sample["csi_amplitude"], 0.5)


def test_sequence_dataset_builds_contiguous_windows(tmp_path) -> None:
    dataset_path = tmp_path / "synthetic.h5"
    _write_synthetic_h5(dataset_path)

    dataset = MMFiPoseSequenceDataset(dataset_path, split="train", window_size=2, window_stride=1)
    sample = dataset[0]

    assert len(dataset) == 4
    assert sample["keypoints"].shape == (2, 17, 2)
    assert sample["csi_amplitude"].shape == (2, 3, 114, 10)
    assert sample["csi_phase_cos"].shape == (2, 3, 114, 10)
    assert sample["frame_id"] == "frame002"


def test_build_h5_dataset_parser_defaults() -> None:
    build_h5_dataset = _load_build_h5_script()
    args = build_h5_dataset.parse_args(["--output-path", "data/mmfi_pose.h5"])

    assert args.dataset_root is None
    assert args.output_path == "data/mmfi_pose.h5"
    assert args.seed == 42
