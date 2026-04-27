from __future__ import annotations

"""HDF5-backed dataloader and raw-dataset packing utilities for MM-Fi pose data."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

try:
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - handled at runtime when torch is unavailable.
    DataLoader = None
    Dataset = object


# Basic parameters setting, containing root path, split setting and shape etc.
DEFAULT_LOCAL_DATASET_ROOT = Path(r"D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset")
DEFAULT_LINUX_DATASET_ROOT = Path("/data/WiFiPose/dataset/dataset")
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_SCHEME = "action_env"
SPLIT_SCHEMES = (DEFAULT_SPLIT_SCHEME, "frame_random")
SPLIT_RATIOS = {"train": 6, "val": 2, "test": 2}
FRAMES_PER_SAMPLE = 297
KEYPOINT_SHAPE = (17, 2)
CSI_SHAPE = (3, 114, 10)
AMPLITUDE_NORMALIZATION_ATTR = "train_global_minmax"
KEYPOINT_NORMALIZATION_ATTR = "train_axis_max"
PHASE_CLEANING_ATTR = "unwrap_subcarrier_detrend_mean"
RAW_STORAGE_ATTR = "cleaned_raw"
NORMALIZED_STORAGE_ATTR = "normalized"


# Sequence-level : Axx/Syy/rgb(wifi-csi) total frames
@dataclass(frozen=True)
class SampleSequence:
    """One sample sequence under Axx/Syy before expanding it into aligned frames."""

    action: str             # Axx  : A01 - A27
    sample: str             # Syy  : S01 - S40
    environment: str        # envz : env1 - env4, mapped from Syy in blocks of ten samples
    rgb_dir: Path
    csi_dir: Path


# Frame-level : Axx/Syy/envz/rgb(wifi-csi)/framexxx
@dataclass(frozen=True)
class FrameRecord:
    """One aligned frame pair consisting of pose labels and CSI measurements."""

    action: str             # Axx  : A01 - A27
    sample: str             # Syy  : S01 - S40
    environment: str        # envz : env1 - env4, mapped from Syy in blocks of ten samples
    frame_stem: str         # frame indice
    keypoint_path: Path
    csi_path: Path


def resolve_dataset_root(dataset_root: Optional[str | Path] = None) -> Path:
    """Resolve the raw MM-Fi dataset root from an override or machine-specific defaults."""

    if dataset_root is not None:
        root = Path(dataset_root)
    elif DEFAULT_LOCAL_DATASET_ROOT.exists():
        root = DEFAULT_LOCAL_DATASET_ROOT
    else:
        root = DEFAULT_LINUX_DATASET_ROOT

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    
    return root


def resolve_h5_dataset_path(dataset_root: str | Path) -> Path:
    """Resolve the prepacked HDF5 dataset path used for training."""

    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"HDF5 dataset does not exist: {dataset_path}")
    if dataset_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"Expected an HDF5 dataset path, got: {dataset_path}")
    return dataset_path


def validate_split_scheme(split_scheme: str) -> str:
    """Validate one supported split scheme name."""

    if split_scheme not in SPLIT_SCHEMES:
        raise ValueError(f"split_scheme must be one of {SPLIT_SCHEMES}, got {split_scheme}")
    return split_scheme


def _resolve_split_ratios(split_ratios: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Return validated split ratios with the expected split keys."""

    ratios = split_ratios or SPLIT_RATIOS
    if tuple(ratios.keys()) != SPLIT_NAMES:
        raise ValueError(f"Split keys must be exactly {SPLIT_NAMES}, got {tuple(ratios.keys())}")
    return ratios


def _split_indices_dataset_name(split: str, split_scheme: str) -> str:
    """Return the HDF5 dataset name for one scheme-specific split index array."""

    return f"{split_scheme}_{split}_indices"


def _split_attr_name(split_scheme: str, attr_name: str) -> str:
    """Return the HDF5 attribute name for one scheme-specific statistic."""

    return f"{split_scheme}_{attr_name}"


def sample_to_environment(sample_name: str) -> str:
    """
    Map sample ids S01-S40 to env1-env4 in blocks of ten samples.
    Each env contains 10 samples, such as S01-S10 -> env1 and so on.
    """

    sample_index = int(sample_name[1:])                 # Syy -> yy -> turn to int
    environment_index = (sample_index - 1) // 10 + 1    # yy: 1-10 -> env1, 11-20 -> env2, 21-30 -> env3, 31-40 -> env4
    return f"env{environment_index}"


def _sorted_dirs(root: Path, prefix: str) -> List[Path]:
    """List prefixed directories in lexicographic order for deterministic traversal."""

    # use prefix to judge the dir
    return sorted( 
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )


def _sorted_files(directory: Path, pattern: str) -> List[Path]:
    """List files in lexicographic order so frame alignment stays deterministic."""

    return sorted(directory.glob(pattern), key=lambda path: path.name)


def discover_sample_sequences(dataset_root: str | Path) -> List[SampleSequence]:
    """Scan the raw dataset root and collect all available Axx/Syy sample sequences."""

    root = resolve_dataset_root(dataset_root)               # dataset root
    sequences: List[SampleSequence] = []                    # Sequence

    for action_dir in _sorted_dirs(root, "A"):              # Axx
        for sample_dir in _sorted_dirs(action_dir, "S"):    # Syy
            rgb_dir = sample_dir / "rgb"                    # rgb, keypoints
            csi_dir = sample_dir / "wifi-csi"               # wifi-csi, CSI amplitude and phase

            if not rgb_dir.is_dir() or not csi_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected aligned rgb and wifi-csi directories under {sample_dir}"
                )

            # All frames, frame001 - frame297
            # collect all sequences, 27 x 40 in total
            sequences.append(   
                SampleSequence(
                    action=action_dir.name,
                    sample=sample_dir.name,
                    environment=sample_to_environment(sample_dir.name),
                    rgb_dir=rgb_dir,
                    csi_dir=csi_dir,
                )
            )

    if not sequences:
        raise ValueError(f"No sample sequences found under {root}")

    return sequences


def build_sample_splits(
    dataset_root: str | Path,
    seed: int = 42,
    split_ratios: Optional[Dict[str, int]] = None,
) -> Dict[str, List[SampleSequence]]:
    """Split each (action, environment) group with a fixed 6:2:2 sample ratio."""

    ratios = _resolve_split_ratios(split_ratios)
    if sum(ratios.values()) != 10:
        raise ValueError("Per-environment split ratios must sum to 10 samples")

    # unordered grouping: (action, environment) -> Syy Sequence(Frame001 -> Frame297)
    grouped_sequences: Dict[Tuple[str, str], List[SampleSequence]] = {}
    for sequence in discover_sample_sequences(dataset_root):
        grouped_sequences.setdefault((sequence.action, sequence.environment), []).append(sequence)

    
    splits: Dict[str, List[SampleSequence]] = {name: [] for name in SPLIT_NAMES}    # split into different name: train, val, test.
    for (action, environment), sequences in sorted(grouped_sequences.items()):      # sorted by the dir name. Each grouped_sequence(action, env) contains 10 samples.
        
        ordered_sequences = sorted(sequences, key=lambda item: item.sample)         # sort the sequences by name, from S01 to S10, or S11 to S20 etc.
        if len(ordered_sequences) != 10:
            raise ValueError(
                f"Expected 10 samples for {action}/{environment}, found {len(ordered_sequences)}"
            )

        # shuffle first, in Axx/Envz dim.
        group_rng = random.Random(f"{seed}:{action}:{environment}")
        shuffled_sequences = ordered_sequences[:]                       # S01 - S10 for example.
        group_rng.shuffle(shuffled_sequences)                           # shuffle the sequence.

        # choose the first 6 for train, next 2 for val, last 2 for test, from shuffled sequences.
        train_end = ratios["train"]
        val_end = train_end + ratios["val"]
        splits["train"].extend(shuffled_sequences[:train_end])          # 1 - 6
        splits["val"].extend(shuffled_sequences[train_end:val_end])     # 7 - 8
        splits["test"].extend(shuffled_sequences[val_end:])             # 9 - 10

    return splits


def build_frame_splits(
    dataset_root: str | Path,
    seed: int = 42,
    split_ratios: Optional[Dict[str, int]] = None,
) -> Dict[str, List[FrameRecord]]:
    """Randomly split frame records without preserving action/environment sequence boundaries."""

    ratios = _resolve_split_ratios(split_ratios)
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    records = expand_frame_records(discover_sample_sequences(dataset_root))
    ordered_records = sorted(
        records,
        key=lambda record: (record.action, record.environment, record.sample, record.frame_stem),
    )
    shuffled_records = ordered_records[:]
    random.Random(seed).shuffle(shuffled_records)

    train_end = len(shuffled_records) * ratios["train"] // total_ratio
    val_end = len(shuffled_records) * (ratios["train"] + ratios["val"]) // total_ratio
    return {
        "train": shuffled_records[:train_end],
        "val": shuffled_records[train_end:val_end],
        "test": shuffled_records[val_end:],
    }


# from sequence to frame, expand the sequence.
def expand_frame_records(sequences: Sequence[SampleSequence]) -> List[FrameRecord]:
    """Expand selected sample sequences into frame-level aligned label/CSI records."""

    records: List[FrameRecord] = []
    for sequence in sequences:
        keypoint_files = _sorted_files(sequence.rgb_dir, "*.npy")
        csi_files = _sorted_files(sequence.csi_dir, "*.mat")

        if len(keypoint_files) != len(csi_files):
            raise ValueError(
                f"Mismatched frame count for {sequence.action}/{sequence.sample}: "
                f"{len(keypoint_files)} labels vs {len(csi_files)} CSI files"
            )

        if len(keypoint_files) != FRAMES_PER_SAMPLE:
            raise ValueError(
                f"Expected {FRAMES_PER_SAMPLE} frames for {sequence.action}/{sequence.sample}, "
                f"found {len(keypoint_files)}"
            )

        # keypoints, CSI files, pairs the data and labels
        for keypoint_path, csi_path in zip(keypoint_files, csi_files):
            # check whether symmetric file name: frame001.npy to frame001.mat.
            if keypoint_path.stem != csi_path.stem:
                raise ValueError(
                    f"Frame mismatch for {sequence.action}/{sequence.sample}: "
                    f"{keypoint_path.name} vs {csi_path.name}"
                )

            # expand to frame-level records, frame001 - frame297
            records.append(
                FrameRecord(
                    action=sequence.action,
                    sample=sequence.sample,
                    environment=sequence.environment,
                    frame_stem=keypoint_path.stem,
                    keypoint_path=keypoint_path,
                    csi_path=csi_path,
                )
            )

    return records


def _decode_string(value: str | bytes) -> str:
    """Normalize HDF5 string values to plain Python strings."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# load the keypoints and amp, not including phase.
def _load_raw_keypoints_and_amplitude(record: FrameRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load one aligned raw frame's labels and CSI amplitude."""

    keypoints = np.load(record.keypoint_path).astype(np.float32)
    csi_data = loadmat(record.csi_path, variable_names=["CSIamp"])
    csi_amplitude = np.asarray(csi_data["CSIamp"], dtype=np.float32)

    if keypoints.shape != KEYPOINT_SHAPE:
        raise ValueError(f"Unexpected keypoint shape for {record.keypoint_path}: {keypoints.shape}")
    if csi_amplitude.shape != CSI_SHAPE:
        raise ValueError(f"Unexpected CSI amplitude shape for {record.csi_path}: {csi_amplitude.shape}")

    return keypoints, csi_amplitude


# load the keypoints and full csi dta, including phase.
def _load_raw_frame(record: FrameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one aligned raw frame and validate its expected shapes."""

    keypoints = np.load(record.keypoint_path).astype(np.float32)
    csi_data = loadmat(record.csi_path, variable_names=["CSIamp", "CSIphase"])
    csi_amplitude = np.asarray(csi_data["CSIamp"], dtype=np.float32)
    csi_phase = np.asarray(csi_data["CSIphase"], dtype=np.float32)

    if keypoints.shape != KEYPOINT_SHAPE:
        raise ValueError(f"Unexpected keypoint shape for {record.keypoint_path}: {keypoints.shape}")
    if csi_amplitude.shape != CSI_SHAPE:
        raise ValueError(f"Unexpected CSI amplitude shape for {record.csi_path}: {csi_amplitude.shape}")
    if csi_phase.shape != CSI_SHAPE:
        raise ValueError(f"Unexpected CSI phase shape for {record.csi_path}: {csi_phase.shape}")

    return keypoints, csi_amplitude, csi_phase


def _validate_keypoints(keypoints: np.ndarray, source: Path) -> np.ndarray:
    """Ensure keypoint labels are finite before normalization and training use."""

    if not np.isfinite(keypoints).all():
        raise ValueError(f"Keypoints contain non-finite values: {source}")
    return keypoints


def _clean_csi_amplitude(csi_amplitude: np.ndarray, source: Path) -> np.ndarray:
    """Replace non-finite amplitude values with finite frame-local bounds."""

    finite_mask = np.isfinite(csi_amplitude)        # find the finite value in csi amplitude, and get its indice mask
    if finite_mask.all():
        return csi_amplitude                        # if all the value is finite, return directly (no infinite or nan values)

    finite_values = csi_amplitude[finite_mask]      # get the valid value based on the mask
    if finite_values.size == 0:
        raise ValueError(f"CSI amplitude contains no finite values: {source}")

    finite_min = np.min(finite_values)              # minimum of the finite value
    finite_max = np.max(finite_values)              # maximum of the finite value
    cleaned = csi_amplitude.copy()
    cleaned[np.isnan(cleaned)] = finite_min         # nan -> minimum
    cleaned[np.isneginf(cleaned)] = finite_min      # -inf -> minimum
    cleaned[np.isposinf(cleaned)] = finite_max      # +inf -> maximum

    if not np.isfinite(cleaned).all():
        raise ValueError(f"CSI amplitude still contains non-finite values after cleaning: {source}")

    return cleaned


def _clean_csi_phase(csi_phase: np.ndarray, source: Path) -> np.ndarray:
    """Unwrap phase and remove per-antenna subcarrier linear trends."""

    cleaned = np.asarray(csi_phase, dtype=np.float32).copy()
    subcarrier_indices = np.arange(CSI_SHAPE[1], dtype=np.float32)

    for antenna_index in range(CSI_SHAPE[0]):                       # each antenna
        for time_index in range(CSI_SHAPE[2]):                      # each time point
            phase_line = cleaned[antenna_index, :, time_index]      # all subcarriers
            finite_mask = np.isfinite(phase_line)                   # find the finite value indice.
            
            if not finite_mask.any():
                raise ValueError(f"CSI phase contains no finite values: {source}")
            if not finite_mask.all():
                phase_line = np.interp(                             # interpolate the phase for infinite subcarrier value
                    subcarrier_indices,
                    subcarrier_indices[finite_mask],
                    phase_line[finite_mask],
                ).astype(np.float32)
                cleaned[antenna_index, :, time_index] = phase_line  # append to cleaned phase

    unwrapped = np.unwrap(cleaned, axis=1).astype(np.float32)       # unwrap the phase to prevent huge jump

    centered_subcarriers = (                                        # x - x_mean
        subcarrier_indices - float(np.mean(subcarrier_indices))     # -56.5, ..., -0.5, 0.5, ..., 56.5
    ).reshape(1, CSI_SHAPE[1], 1)                                   # [1, 114, 1]

    phase_mean = np.mean(unwrapped, axis=1, keepdims=True)          # calculate the mean of each antenna and time slot, 3 x 1 x 10
    
    
    slope = np.sum(centered_subcarriers * (unwrapped - phase_mean), axis=1, keepdims=True)  # sum( (x - x_mean) * (y - y_mean) )
    slope = slope / np.sum(centered_subcarriers * centered_subcarriers)                     # slope / sum( (x - x_mean)^2 )
    linear_trend = slope * centered_subcarriers + phase_mean                                # linear = slope * (x - x_mean) + y_mean
    calibrated = (unwrapped - linear_trend).astype(np.float32)

    if not np.isfinite(calibrated).all():
        raise ValueError(f"CSI phase still contains non-finite values after cleaning: {source}")

    return calibrated


def _compute_csi_phase_cos(csi_phase: np.ndarray) -> np.ndarray:
    """Convert cleaned phase to a bounded cosine feature."""

    return np.cos(csi_phase).astype(np.float32)


def _prepare_keypoints_and_amplitude(record: FrameRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load one frame's labels and amplitude for split-level statistics."""

    keypoints, csi_amplitude = _load_raw_keypoints_and_amplitude(record)
    keypoints = _validate_keypoints(keypoints, source=record.keypoint_path)
    csi_amplitude = _clean_csi_amplitude(csi_amplitude, source=record.csi_path)
    return keypoints, csi_amplitude


def _prepare_raw_frame(record: FrameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one frame and apply the cleaning required for stable training."""

    keypoints, csi_amplitude, csi_phase = _load_raw_frame(record)                   # get keypoint and data in raw format
    keypoints = _validate_keypoints(keypoints, source=record.keypoint_path)         # validate
    csi_amplitude = _clean_csi_amplitude(csi_amplitude, source=record.csi_path)     # clean
    csi_phase = _clean_csi_phase(csi_phase, source=record.csi_path)
    csi_phase_cos = _compute_csi_phase_cos(csi_phase)
    return keypoints, csi_amplitude, csi_phase, csi_phase_cos


def _compute_train_amplitude_bounds(records: Sequence[FrameRecord]) -> tuple[float, float]:
    """Compute global min/max on cleaned train-split amplitudes only."""

    global_min = float("inf")       # global minimum
    global_max = float("-inf")      # global maximum

    # find by frame
    for record in tqdm(records, desc="Computing train amplitude bounds", dynamic_ncols=True):
        _, csi_amplitude = _prepare_keypoints_and_amplitude(record)
        global_min = min(global_min, float(np.min(csi_amplitude)))
        global_max = max(global_max, float(np.max(csi_amplitude)))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("Train amplitude bounds must be finite after cleaning")
    if global_max <= global_min:
        raise ValueError(
            f"Train amplitude bounds are invalid for min-max normalization: min={global_min}, max={global_max}"
        )

    return global_min, global_max


def _normalize_csi_amplitude(
    csi_amplitude: np.ndarray,
    train_min: float,
    train_max: float,
) -> np.ndarray:
    """Apply one train-split global min-max normalization to all amplitudes."""

    normalized = (csi_amplitude - train_min) / (train_max - train_min)
    return normalized.astype(np.float32)


def _compute_train_keypoint_scales(records: Sequence[FrameRecord]) -> tuple[float, float]:
    """Compute per-axis global maxima on the train split for stable keypoint scaling."""

    global_x_max = float("-inf")
    global_y_max = float("-inf")

    for record in tqdm(records, desc="Computing train keypoint scales", dynamic_ncols=True):
        keypoints, _ = _prepare_keypoints_and_amplitude(record)
        global_x_max = max(global_x_max, float(np.max(keypoints[:, 0])))
        global_y_max = max(global_y_max, float(np.max(keypoints[:, 1])))

    if not np.isfinite(global_x_max) or not np.isfinite(global_y_max):
        raise ValueError("Train keypoint scales must be finite")
    if global_x_max <= 0.0 or global_y_max <= 0.0:
        raise ValueError(
            f"Train keypoint scales must be positive, got x={global_x_max}, y={global_y_max}"
        )

    return global_x_max, global_y_max


def _normalize_keypoints(
    keypoints: np.ndarray,
    x_scale: float,
    y_scale: float,
) -> np.ndarray:
    """Apply one train-split global axis-wise scaling to keypoint coordinates."""

    normalized = keypoints.copy()
    normalized[:, 0] = normalized[:, 0] / x_scale
    normalized[:, 1] = normalized[:, 1] / y_scale
    return normalized.astype(np.float32)


# de-normalize the keypoints for visualization.
def denormalize_keypoints(
    keypoints: np.ndarray,
    x_scale: float,
    y_scale: float,
) -> np.ndarray:
    """Restore normalized keypoints to their original pixel-coordinate scale."""

    restored = np.asarray(keypoints, dtype=np.float32).copy()
    restored[..., 0] = restored[..., 0] * x_scale
    restored[..., 1] = restored[..., 1] * y_scale
    return restored.astype(np.float32)


def build_h5_dataset(
    dataset_root: str | Path,
    output_path: str | Path,
    seed: int = 42,
    split_ratios: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Pack the raw MM-Fi dataset into a single HDF5 file with materialized split indices."""

    source_root = resolve_dataset_root(dataset_root)        # raw dataset root
    target_path = Path(output_path)                         # target hdf5 dataset path
    target_path.parent.mkdir(parents=True, exist_ok=True)   # ensure the dir exists

    all_sequences = discover_sample_sequences(source_root)
    all_records = expand_frame_records(all_sequences)
    record_to_index = {record: index for index, record in enumerate(all_records)}

    sample_splits = build_sample_splits(source_root, seed=seed, split_ratios=split_ratios)
    split_records_by_scheme = {
        DEFAULT_SPLIT_SCHEME: {
            split_name: expand_frame_records(sample_splits[split_name]) for split_name in SPLIT_NAMES
        },
        "frame_random": build_frame_splits(source_root, seed=seed, split_ratios=split_ratios),
    }

    scheme_stats = {}
    for split_scheme, split_records in split_records_by_scheme.items():
        train_records = split_records["train"]
        train_min, train_max = _compute_train_amplitude_bounds(train_records)
        keypoint_x_scale, keypoint_y_scale = _compute_train_keypoint_scales(train_records)
        scheme_stats[split_scheme] = {
            "amplitude_train_min": train_min,
            "amplitude_train_max": train_max,
            "keypoint_x_scale": keypoint_x_scale,
            "keypoint_y_scale": keypoint_y_scale,
        }
    default_scheme_stats = scheme_stats[DEFAULT_SPLIT_SCHEME]
    
    # calculate the size of each dataset split, pre-allocated.
    total_records = len(all_records)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    # "w" -> write
    with h5py.File(target_path, "w") as h5_file:
        
        # allocate datasets for keypoints, amp, phase and metadata.
        keypoints_dataset = h5_file.create_dataset(
            "keypoints", shape=(total_records, *KEYPOINT_SHAPE), dtype=np.float32
        )
        amplitude_dataset = h5_file.create_dataset(
            "csi_amplitude", shape=(total_records, *CSI_SHAPE), dtype=np.float32
        )
        phase_dataset = h5_file.create_dataset(
            "csi_phase", shape=(total_records, *CSI_SHAPE), dtype=np.float32
        )
        phase_cos_dataset = h5_file.create_dataset(
            "csi_phase_cos", shape=(total_records, *CSI_SHAPE), dtype=np.float32
        )
        action_dataset = h5_file.create_dataset("action", shape=(total_records,), dtype=string_dtype)
        sample_dataset = h5_file.create_dataset("sample", shape=(total_records,), dtype=string_dtype)
        environment_dataset = h5_file.create_dataset(
            "environment", shape=(total_records,), dtype=string_dtype
        )
        frame_dataset = h5_file.create_dataset("frame_id", shape=(total_records,), dtype=string_dtype)

        h5_file.attrs["source_root"] = str(source_root)
        h5_file.attrs["seed"] = seed
        h5_file.attrs["frames_per_sample"] = FRAMES_PER_SAMPLE
        h5_file.attrs["storage_format"] = RAW_STORAGE_ATTR
        h5_file.attrs["default_split_scheme"] = DEFAULT_SPLIT_SCHEME
        h5_file.attrs["available_split_schemes"] = ",".join(SPLIT_SCHEMES)
        h5_file.attrs["amplitude_normalization"] = AMPLITUDE_NORMALIZATION_ATTR
        h5_file.attrs["amplitude_train_min"] = default_scheme_stats["amplitude_train_min"]
        h5_file.attrs["amplitude_train_max"] = default_scheme_stats["amplitude_train_max"]
        h5_file.attrs["keypoint_normalization"] = KEYPOINT_NORMALIZATION_ATTR
        h5_file.attrs["keypoint_x_scale"] = default_scheme_stats["keypoint_x_scale"]
        h5_file.attrs["keypoint_y_scale"] = default_scheme_stats["keypoint_y_scale"]
        h5_file.attrs["phase_cleaning"] = PHASE_CLEANING_ATTR
        for split_scheme, stats in scheme_stats.items():
            for name, value in stats.items():
                h5_file.attrs[_split_attr_name(split_scheme, name)] = value

        # Start writing the data
        with tqdm(total=total_records, desc="Packing HDF5", dynamic_ncols=True) as progress_bar:
            for dataset_index, record in enumerate(all_records):
                keypoints, csi_amplitude, csi_phase, csi_phase_cos = _prepare_raw_frame(record)

                keypoints_dataset[dataset_index] = keypoints                        # write cleaned raw data once
                amplitude_dataset[dataset_index] = csi_amplitude
                phase_dataset[dataset_index] = csi_phase
                phase_cos_dataset[dataset_index] = csi_phase_cos
                action_dataset[dataset_index] = record.action
                sample_dataset[dataset_index] = record.sample
                environment_dataset[dataset_index] = record.environment
                frame_dataset[dataset_index] = record.frame_stem
                progress_bar.update(1)

        for split_name in SPLIT_NAMES:
            default_indices = np.asarray(
                [record_to_index[record] for record in split_records_by_scheme[DEFAULT_SPLIT_SCHEME][split_name]],
                dtype=np.int64,
            )
            h5_file.create_dataset(f"{split_name}_indices", data=default_indices)

        for split_scheme, split_records in split_records_by_scheme.items():
            for split_name in SPLIT_NAMES:
                indices = np.asarray(
                    [record_to_index[record] for record in split_records[split_name]],
                    dtype=np.int64,
                )
                h5_file.create_dataset(_split_indices_dataset_name(split_name, split_scheme), data=indices)

    return {
        "num_records": total_records,
        "num_train_frames": len(split_records_by_scheme[DEFAULT_SPLIT_SCHEME]["train"]),
        "num_val_frames": len(split_records_by_scheme[DEFAULT_SPLIT_SCHEME]["val"]),
        "num_test_frames": len(split_records_by_scheme[DEFAULT_SPLIT_SCHEME]["test"]),
    }


class MMFiPoseDataset(Dataset):
    """Frame-level HDF5 dataset that returns aligned pose labels and CSI tensors."""

    def __init__(self, dataset_root: str | Path, split: str, split_scheme: str = DEFAULT_SPLIT_SCHEME) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split}")

        self.dataset_root = resolve_h5_dataset_path(dataset_root)
        self.split = split
        self.split_scheme = validate_split_scheme(split_scheme)
        self._h5_file: h5py.File | None = None

        with h5py.File(self.dataset_root, "r") as h5_file:
            index_dataset_name = _split_indices_dataset_name(split, self.split_scheme)
            if index_dataset_name in h5_file:
                self.indices = np.asarray(h5_file[index_dataset_name], dtype=np.int64)
            elif self.split_scheme == DEFAULT_SPLIT_SCHEME and f"{split}_indices" in h5_file:
                self.indices = np.asarray(h5_file[f"{split}_indices"], dtype=np.int64)
            else:
                raise KeyError(
                    f"HDF5 dataset does not contain indices for split_scheme={self.split_scheme!r}, split={split!r}. "
                    "Rebuild the dataset with dual split support."
                )
            self.storage_format = _decode_string(h5_file.attrs.get("storage_format", NORMALIZED_STORAGE_ATTR))
            self.keypoint_normalization = _decode_string(
                h5_file.attrs.get("keypoint_normalization", "")
            )
            self.amplitude_normalization = _decode_string(
                h5_file.attrs.get("amplitude_normalization", "")
            )
            self.keypoint_x_scale = float(
                h5_file.attrs.get(
                    _split_attr_name(self.split_scheme, "keypoint_x_scale"),
                    h5_file.attrs.get("keypoint_x_scale", 1.0),
                )
            )
            self.keypoint_y_scale = float(
                h5_file.attrs.get(
                    _split_attr_name(self.split_scheme, "keypoint_y_scale"),
                    h5_file.attrs.get("keypoint_y_scale", 1.0),
                )
            )
            self.amplitude_train_min = float(
                h5_file.attrs.get(
                    _split_attr_name(self.split_scheme, "amplitude_train_min"),
                    h5_file.attrs.get("amplitude_train_min", 0.0),
                )
            )
            self.amplitude_train_max = float(
                h5_file.attrs.get(
                    _split_attr_name(self.split_scheme, "amplitude_train_max"),
                    h5_file.attrs.get("amplitude_train_max", 1.0),
                )
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _get_h5_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.dataset_root, "r")
        return self._h5_file

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup.
        self.close()

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | str]:
        """Load one frame's normalized keypoints and CSI amplitude arrays from HDF5."""

        h5_file = self._get_h5_file()
        frame_index = int(self.indices[index])
        keypoints = np.asarray(h5_file["keypoints"][frame_index], dtype=np.float32)
        csi_amplitude = np.asarray(h5_file["csi_amplitude"][frame_index], dtype=np.float32)
        if self.storage_format == RAW_STORAGE_ATTR:
            keypoints = _normalize_keypoints(
                keypoints,
                x_scale=self.keypoint_x_scale,
                y_scale=self.keypoint_y_scale,
            )
            csi_amplitude = _normalize_csi_amplitude(
                csi_amplitude,
                train_min=self.amplitude_train_min,
                train_max=self.amplitude_train_max,
            )

        return {
            "action": _decode_string(h5_file["action"][frame_index]),
            "sample": _decode_string(h5_file["sample"][frame_index]),
            "environment": _decode_string(h5_file["environment"][frame_index]),
            "frame_id": _decode_string(h5_file["frame_id"][frame_index]),
            "keypoints": keypoints,
            "csi_amplitude": csi_amplitude,
            "csi_phase": np.asarray(h5_file["csi_phase"][frame_index], dtype=np.float32),
            "csi_phase_cos": np.asarray(h5_file["csi_phase_cos"][frame_index], dtype=np.float32),
        }


def _frame_number(frame_id: str) -> int:
    """Extract the numeric part of a frame id such as ``frame001``."""

    digits = "".join(character for character in frame_id if character.isdigit())
    if not digits:
        raise ValueError(f"Frame id does not contain a numeric suffix: {frame_id}")
    return int(digits)


class MMFiPoseSequenceDataset(Dataset):
    """Windowed HDF5 dataset for sequence-level pose regression.

    The packed HDF5 file is frame-level. This wrapper builds contiguous windows
    from frames that share the same action, sample, and environment, then loads
    the CSI and pose tensors for one full window at a time.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        split_scheme: str = DEFAULT_SPLIT_SCHEME,
        window_size: int = 16,
        window_stride: int = 4,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if window_stride <= 0:
            raise ValueError(f"window_stride must be positive, got {window_stride}")

        self.frame_dataset = MMFiPoseDataset(
            dataset_root=dataset_root,
            split=split,
            split_scheme=split_scheme,
        )
        self.dataset_root = self.frame_dataset.dataset_root
        self.split = split
        self.split_scheme = split_scheme
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.keypoint_x_scale = self.frame_dataset.keypoint_x_scale
        self.keypoint_y_scale = self.frame_dataset.keypoint_y_scale
        self.windows = self._build_windows()

        if not self.windows:
            raise ValueError(
                "No contiguous sequence windows were found. "
                f"split={split!r}, split_scheme={split_scheme!r}, "
                f"window_size={window_size}, window_stride={window_stride}"
            )

    def _build_windows(self) -> list[tuple[int, ...]]:
        """Build contiguous windows without crossing sample boundaries."""

        grouped_frames: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        with h5py.File(self.dataset_root, "r") as h5_file:
            # Read string metadata once. Per-frame HDF5 string access is slow on
            # large packed datasets and can delay training startup noticeably.
            actions = np.asarray(h5_file["action"])
            samples = np.asarray(h5_file["sample"])
            environments = np.asarray(h5_file["environment"])
            frame_ids = np.asarray(h5_file["frame_id"])
            for frame_index in self.frame_dataset.indices:
                frame_index = int(frame_index)
                action = _decode_string(actions[frame_index])
                sample = _decode_string(samples[frame_index])
                environment = _decode_string(environments[frame_index])
                frame_id = _decode_string(frame_ids[frame_index])
                group_key = (action, sample, environment)
                grouped_frames.setdefault(group_key, []).append((_frame_number(frame_id), frame_index))

        windows: list[tuple[int, ...]] = []
        for frames in grouped_frames.values():
            ordered_frames = sorted(frames, key=lambda item: item[0])
            frame_numbers = [frame_number for frame_number, _ in ordered_frames]
            frame_indices = [frame_index for _, frame_index in ordered_frames]

            for start in range(0, len(frame_indices) - self.window_size + 1, self.window_stride):
                number_slice = frame_numbers[start : start + self.window_size]
                expected = list(range(number_slice[0], number_slice[0] + self.window_size))
                if number_slice == expected:
                    windows.append(tuple(frame_indices[start : start + self.window_size]))

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["frame_dataset"] = self.frame_dataset.__getstate__()
        return state

    def __setstate__(self, state: dict) -> None:
        frame_dataset_state = state.pop("frame_dataset")
        self.__dict__.update(state)
        self.frame_dataset = MMFiPoseDataset(
            dataset_root=self.dataset_root,
            split=self.split,
            split_scheme=self.split_scheme,
        )
        self.frame_dataset.__dict__.update(frame_dataset_state)

    def close(self) -> None:
        self.frame_dataset.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup.
        self.close()

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | str]:
        """Load one contiguous sequence window."""

        h5_file = self.frame_dataset._get_h5_file()
        window_indices = self.windows[index]
        keypoints: list[np.ndarray] = []
        csi_amplitude: list[np.ndarray] = []
        csi_phase_cos: list[np.ndarray] = []

        for frame_index in window_indices:
            frame_keypoints = np.asarray(h5_file["keypoints"][frame_index], dtype=np.float32)
            frame_amplitude = np.asarray(h5_file["csi_amplitude"][frame_index], dtype=np.float32)
            if self.frame_dataset.storage_format == RAW_STORAGE_ATTR:
                frame_keypoints = _normalize_keypoints(
                    frame_keypoints,
                    x_scale=self.keypoint_x_scale,
                    y_scale=self.keypoint_y_scale,
                )
                frame_amplitude = _normalize_csi_amplitude(
                    frame_amplitude,
                    train_min=self.frame_dataset.amplitude_train_min,
                    train_max=self.frame_dataset.amplitude_train_max,
                )

            keypoints.append(frame_keypoints)
            csi_amplitude.append(frame_amplitude)
            csi_phase_cos.append(np.asarray(h5_file["csi_phase_cos"][frame_index], dtype=np.float32))

        middle_index = window_indices[len(window_indices) // 2]
        return {
            "action": _decode_string(h5_file["action"][middle_index]),
            "sample": _decode_string(h5_file["sample"][middle_index]),
            "environment": _decode_string(h5_file["environment"][middle_index]),
            "frame_id": _decode_string(h5_file["frame_id"][middle_index]),
            "keypoints": np.stack(keypoints).astype(np.float32),
            "csi_amplitude": np.stack(csi_amplitude).astype(np.float32),
            "csi_phase_cos": np.stack(csi_phase_cos).astype(np.float32),
        }


# Put MMFiPoseDataset into DataLoader, batch and shuffle.
def create_data_loader(
    dataset_root: str | Path,
    split: str,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    split_scheme: str = DEFAULT_SPLIT_SCHEME,
    split_ratios: Optional[Dict[str, int]] = None,
):
    """Create one PyTorch DataLoader for the requested HDF5 split."""

    if DataLoader is None:
        raise ImportError(
            "PyTorch is not installed in the current environment. "
            "Install torch to create DataLoader instances."
        )

    del seed, split_ratios  # pre-allocated while build .h5 file

    dataset = MMFiPoseDataset(dataset_root=dataset_root, split=split, split_scheme=split_scheme)
    should_shuffle = shuffle if shuffle is not None else split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
    )


def create_data_loaders(
    dataset_root: str | Path,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    split_scheme: str = DEFAULT_SPLIT_SCHEME,
    split_ratios: Optional[Dict[str, int]] = None,
):
    """Create train/val/test DataLoaders from the same HDF5 dataset."""

    return {
        split: create_data_loader(
            dataset_root=dataset_root,
            split=split,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            split_scheme=split_scheme,
            split_ratios=split_ratios,
        )
        for split in SPLIT_NAMES
    }


# ========== CLI utilities for dataset inspection and debugging ==========

def summarize_splits(
    dataset_root: str | Path,
    split_scheme: str = DEFAULT_SPLIT_SCHEME,
) -> Dict[str, Dict[str, int]]:
    """Return split statistics for the prepacked HDF5 dataset."""

    dataset_path = resolve_h5_dataset_path(dataset_root)
    validated_split_scheme = validate_split_scheme(split_scheme)
    summary: Dict[str, Dict[str, int]] = {}

    with h5py.File(dataset_path, "r") as h5_file:
        action_dataset = h5_file["action"]
        environment_dataset = h5_file["environment"]

        for split_name in SPLIT_NAMES:
            dataset_name = _split_indices_dataset_name(split_name, validated_split_scheme)
            if dataset_name in h5_file:
                indices = np.asarray(h5_file[dataset_name], dtype=np.int64)
            elif validated_split_scheme == DEFAULT_SPLIT_SCHEME and f"{split_name}_indices" in h5_file:
                indices = np.asarray(h5_file[f"{split_name}_indices"], dtype=np.int64)
            else:
                raise KeyError(
                    f"HDF5 dataset does not contain indices for split_scheme={validated_split_scheme!r}, split={split_name!r}. "
                    "Rebuild the dataset with dual split support."
                )
            actions = {_decode_string(action_dataset[index]) for index in indices}
            environments = {_decode_string(environment_dataset[index]) for index in indices}
            summary[split_name] = {
                "num_frames": len(indices),
                "num_actions": len(actions),
                "num_environments": len(environments),
            }

    return summary


def _preview_sample(dataset: MMFiPoseDataset) -> Dict[str, Tuple[int, ...] | str]:
    """Load the first sample of a split and expose only shape-level information."""

    sample = dataset[0]
    return {
        "action": sample["action"],
        "sample": sample["sample"],
        "environment": sample["environment"],
        "frame_id": sample["frame_id"],
        "keypoints_shape": tuple(sample["keypoints"].shape),
        "csi_amplitude_shape": tuple(sample["csi_amplitude"].shape),
        "csi_phase_shape": tuple(sample["csi_phase"].shape),
        "csi_phase_cos_shape": tuple(sample["csi_phase_cos"].shape),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for HDF5 split summary and optional sample preview."""

    parser = argparse.ArgumentParser(description="MM-Fi HDF5 dataloader preview")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument(
        "--split-scheme",
        type=str,
        default=DEFAULT_SPLIT_SCHEME,
        choices=SPLIT_SCHEMES,
        help="Which split scheme to inspect",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Load one sample from each split and print its shapes",
    )
    return parser.parse_args()


def main() -> None:
    """Print HDF5 split statistics and, optionally, one loaded sample per split."""

    args = parse_args()
    dataset_path = resolve_h5_dataset_path(args.dataset_root)
    summary = summarize_splits(dataset_path, split_scheme=args.split_scheme)

    print(f"dataset_root: {dataset_path}")
    print(f"split_scheme: {args.split_scheme}")
    for split_name in SPLIT_NAMES:
        split_info = summary[split_name]
        print(
            f"{split_name}: frames={split_info['num_frames']}, "
            f"actions={split_info['num_actions']}, "
            f"environments={split_info['num_environments']}"
        )

    if args.preview:
        for split_name in SPLIT_NAMES:
            dataset = MMFiPoseDataset(dataset_root=dataset_path, split=split_name, split_scheme=args.split_scheme)
            print(f"{split_name}_preview: {_preview_sample(dataset)}")
            dataset.close()


if __name__ == "__main__":
    main()
