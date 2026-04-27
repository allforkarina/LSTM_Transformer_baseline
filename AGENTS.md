# Repository Guidelines

## Project Structure & Module Organization
- `dataloader.py`: Core module for discovering raw MM-Fi samples, validating frame alignment, building split indices, cleaning CSI features, packing one HDF5 dataset, loading HDF5 splits, creating PyTorch `DataLoader` instances, and previewing split contents.
- `baseline_common.py`: Shared training and evaluation helpers such as sequence loss computation, pixel-space metrics, checkpoint config handling, dataloader construction, and device selection.
- `models/`: Baseline model package. The current baseline is a sequence-level CSI pose regressor with a frame CNN encoder, a 4-layer Transformer temporal encoder, and a joint-query pose decoder.
- `train.py`: Training entrypoint for Linux-server experiments, including `tqdm` epoch visualization, readable CSV logging, structured pose/bone/temporal losses, AdamW with cosine annealing, gradient clipping, early stopping, checkpointing, and test-set evaluation after the best validation checkpoint is selected.
- `eval.py`: Evaluation entrypoint for loading one checkpoint, computing pixel-space test metrics, and saving middle-frame CSI/skeleton visualizations for each `(action, environment)` group in the evaluated split.
- `scripts/build_h5_dataset.py`: Command-line wrapper that packs the raw MM-Fi directory tree into one `.h5` or `.hdf5` dataset file.
- `tests/`: `pytest` tests for baseline model shape checks and CLI parsing.
- `.gitignore`: Excludes Python caches, pytest caches, local packed data, and generated experiment outputs.

This repository is a small baseline subproject focused on code authoring on the local machine and training or evaluation execution on a Linux server. The local machine should only modify and verify code lightly; heavy training, validation, testing runs, and result generation belong on Linux.

Generated datasets can be large and should not be committed. Keep raw dataset roots outside the repository.

## Dataset Format & Physical Features
- Raw data is expected under `Axx/Syy/`, where each sample directory contains both `rgb/` and `wifi-csi/`.
- `rgb/` stores pose labels as `*.npy` files. Each frame must have shape `(17, 2)` for 17 two-dimensional keypoints.
- `wifi-csi/` stores CSI files as `*.mat` files. Each aligned frame must contain `CSIamp` and `CSIphase`, both with shape `(3, 114, 10)`.
- The CSI tensor represents 3 antenna or spatial channels, 114 subcarriers, and 10 time samples per frame.
- Each `Axx/Syy` sequence must contain exactly 297 aligned frames, and frame stems must match one-to-one, such as `frame001.npy` with `frame001.mat`.
- `environment` is derived from the sample id: `S01-S10 -> env1`, `S11-S20 -> env2`, `S21-S30 -> env3`, `S31-S40 -> env4`.

The repository currently uses these physical features:
- `csi_amplitude`: cleaned CSI amplitude. Non-finite values are replaced with frame-local finite bounds, then normalized with train-split global min-max statistics when loaded from raw-storage HDF5.
- `csi_phase`: cleaned CSI phase. The pipeline interpolates non-finite subcarrier values, unwraps phase along the subcarrier axis, and removes the per-antenna linear trend.
- `csi_phase_cos`: cosine-transformed cleaned phase for a bounded phase feature.

The current baseline model uses `csi_amplitude` and `csi_phase_cos` as inputs. It groups frame-level HDF5 records into contiguous windows and predicts normalized `(17, 2)` keypoints for every frame in the window.

One packed HDF5 stores:
- frame-level `keypoints`, `csi_amplitude`, `csi_phase`, `csi_phase_cos`
- metadata `action`, `sample`, `environment`, `frame_id`
- split indices for both `action_env` and `frame_random`
- train-split normalization statistics in HDF5 attributes

## Build, Inspection, and Development Commands
Use the existing Conda environment for project commands:

```powershell
conda activate WiFiPose
```

Build one HDF5 dataset from the raw MM-Fi directory tree:

```powershell
python scripts\build_h5_dataset.py --dataset-root D:\path\to\raw\dataset --output-path data\mmfi_pose.h5 --seed 42
```

Inspect one packed HDF5 dataset and preview sample shapes:

```powershell
python dataloader.py --dataset-root data\mmfi_pose.h5 --split-scheme action_env --preview
```

Inspect the alternate frame-random split scheme:

```powershell
python dataloader.py --dataset-root data\mmfi_pose.h5 --split-scheme frame_random --preview
```

Train the baseline model on Linux with the stricter sample-level split:

```powershell
python train.py --dataset-root /data/WiFiPose/dataset/mmfi_pose.h5 --split-scheme action_env --window-size 16 --window-stride 4 --epochs 60 --batch-size 64 --output-dir outputs/action_env_transformer_sequence
```

Train the baseline model on Linux with the frame-random comparison split:

```powershell
python train.py --dataset-root /data/WiFiPose/dataset/mmfi_pose.h5 --split-scheme frame_random --window-size 16 --window-stride 4 --epochs 60 --batch-size 64 --output-dir outputs/frame_random_transformer_sequence
```

Evaluate one checkpoint and save metrics plus per-`(action, environment)` visualizations:

```powershell
python eval.py --dataset-root /data/WiFiPose/dataset/mmfi_pose.h5 --checkpoint outputs/action_env_transformer_sequence/best_val_mpjpe.pth --output-dir outputs/eval_action_env_transformer_sequence
```

The evaluation script writes one `.png` file for each `(action, environment)` combination covered by the evaluated split. For the standard MM-Fi `action_env` setup this means `27 x 4 = 108` visualization images under `output_dir/visualizations/`.

Training logs are written to `train_log.csv` with one row per epoch. The log includes grouped train and validation loss parts, pixel-space `MPJPE`, pixel-space `PCK@10px`, `PCK@20px`, `PCK@30px`, elapsed time, the current best validation MPJPE, an `is_best` marker, and the current early-stopping counter.

## Current Baseline Design
- Input shape: `B x T x 2 x 3 x 114 x 10`, where the two modalities are normalized amplitude and cosine-transformed phase.
- Default sequence window: `T=16`, `window_stride=4`. Windows never cross `(action, sample, environment)` boundaries.
- Frame CSI encoder: three `Conv2d + BatchNorm2d + ReLU` blocks over `subcarrier x CSI time shot`, followed by global average pooling and a linear projection.
- Temporal encoder: 4-layer Transformer Encoder with 8 attention heads and sinusoidal position encoding over the frame window.
- Pose decoder: 17 learnable COCO joint queries, one per keypoint, concatenated with each temporal feature before shared MLP coordinate regression.
- Output shape: `B x T x 17 x 2` in the existing `[0, 1]` keypoint normalization space.
- Loss: `SmoothL1` pose loss plus `0.1` bone-length loss and `0.05` temporal-delta loss.
- Optimizer and scheduler: AdamW with `CosineAnnealingLR`; global gradient clipping defaults to `--max-grad-norm 1.0` and can be disabled with `--max-grad-norm 0`.
- Metrics: pixel-space `MPJPE` and pixel-space PCK thresholds. Do not use the old normalized-coordinate `PCK@0.05/0.10/0.20` for reporting.

Run lightweight local verification:

```powershell
pytest
```

## Coding Style & Naming Conventions
Use Python 3.10+ syntax, type hints, and `pathlib.Path` for paths. Group imports as standard library, third-party, then local. Follow existing naming: `snake_case` functions and variables, `PascalCase` classes, and uppercase constants such as `SPLIT_NAMES` or `CSI_SHAPE`. Use 4-space indentation. Keep comments focused on dataset assumptions, tensor shapes, alignment constraints, and normalization or cleaning behavior.

## Testing Guidelines
Automated tests use `pytest`. Keep fixtures synthetic and small on the local machine. Prioritize coverage for baseline model output shapes, CLI parsing, raw-directory discovery, frame-count validation, frame-name alignment, HDF5 round-tripping, split generation, shape validation, and CSI cleaning or normalization edge cases.

Local verification should stay lightweight. Do not run long training jobs on the local machine. Use the Linux server for full training, validation, evaluation, and artifact generation.

## Commit & Pull Request Guidelines
Use concise imperative commits, for example `Update AGENTS for baseline dataset project`. Pull requests should include a summary, commands run, dataset assumptions, and any relevant tensor-shape or frame-count output. Do not commit generated datasets, virtual environments, or machine-specific paths.

## Security & Configuration Tips
Do not hard-code private dataset locations beyond documented defaults. Pass dataset paths with `--dataset-root` and keep large or sensitive data outside version control. Prefer Linux-server paths in training and evaluation commands, and keep local Windows paths out of committed experiment configs when possible.

## Agent-Specific Instructions
Write repository-facing agent notes, documentation, and code comments in English. Keep comments aligned with surrounding style. Use Chinese for conversational replies unless the user requests another language.

Whenever project changes affect commands, structure, conventions, testing, configuration, or agent workflow, update this `AGENTS.md` file in the same turn.

After each project modification, commit the change and push it to the configured GitHub remote in the same turn unless the user explicitly asks not to push.

Before changing code, apply the `karpathy-guidelines` skill: state assumptions when needed, prefer the smallest working change, avoid unrelated refactors, and verify the result with a concrete check.

Before running project code or tests, activate the existing Conda environment with `conda activate WiFiPose` to ensure commands run in the established project environment.
