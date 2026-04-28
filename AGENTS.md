# Repository Guidelines

## Project Structure & Module Organization
- `dataloader.py`: Core module for discovering raw MM-Fi samples, validating frame alignment, building split indices, cleaning CSI features, packing one HDF5 dataset, loading HDF5 splits, creating PyTorch `DataLoader` instances, and previewing split contents.
- `models/csi2pose.py`: CSI2Pose model family that encodes CSI windows with a shared per-frame CNN and temporal TCN backbone, then predicts COCO17 keypoints with either a heatmap decoder or a direct-regression decoder.
- `metrics.py`: PCK@5/10/20/50 computation in restored pixel-coordinate space, including overall and per-joint COCO17 scores.
- `train.py`: Server-side training entrypoint with tqdm progress, AMP support, best-checkpoint selection by validation PCK@20, and final test evaluation.
- `test.py`: Server-side checkpoint evaluation entrypoint for train, val, or test splits, with optional GT/prediction pose visualization.
- `visualization.py`: Server-side pose comparison utilities that sample one representative frame per action/environment/sample group and save GT, prediction, and overlay plots in pixel-coordinate space.
- `configs/csi2pose_tcn.yaml`: Default CSI2Pose experiment configuration with decoder-specific settings for `heatmap` and `regression`.
- `scripts/build_h5_dataset.py`: Command-line wrapper that packs the raw MM-Fi directory tree into one `.h5` or `.hdf5` dataset file.
- `tests/`: `pytest` tests for HDF5-backed dataset loading, sequence-window construction, decoder forward shapes, heatmap target construction, decoder-specific losses, skeleton prior losses, PCK metrics, visualization helpers, and CLI parsing.
- `.gitignore`: Excludes Python caches, pytest caches, local packed data, and generated experiment outputs.

This repository is now a small MM-Fi CSI-to-pose baseline project. It contains data preparation/loading utilities plus reproducible CSI2Pose training baselines. The training and test entrypoints support `--decoder heatmap` for the historical heatmap baseline and `--decoder regression` for the current direct-regression baseline. Both decoders use the same CSI backbone and COCO17 joint-query features, and both include train-split skeleton prior losses for body-bone length ratios and joint-angle cosines.

Use the current local checkout for code modification only. Do not run training, evaluation, long data packing, visualization generation, checkpoint export, or any experiment-output generation on this machine. All training runs and all generated outputs must be produced on the Linux server after the server pulls the latest code with `git pull`.

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

The dataset loaders expose both frame-level records and contiguous sequence windows. Sequence-window construction reads HDF5 metadata in bulk before grouping frames, because per-frame HDF5 string reads are too slow for the full packed dataset. CSI2Pose decoder models consume sequence windows and predict every frame in the window; downstream visualization should use the middle frame when only one representative pose is needed.

One packed HDF5 stores:
- frame-level `keypoints`, `csi_amplitude`, `csi_phase`, `csi_phase_cos`
- metadata `action`, `sample`, `environment`, `frame_id`
- split indices for both `action_env` and `frame_random`
- train-split normalization statistics in HDF5 attributes

Use `action_env` as the primary experimental split for reproducible baseline reporting. The `frame_random` split is available for inspection or sanity checks, but it should not be treated as the main generalization result because it can mix frames from the same sequence across splits.

Keypoint normalization intentionally uses train-split coordinate scales stored in the HDF5 attributes. Do not replace this with image-size constants or validation/test-derived statistics unless a new experiment design explicitly changes the normalization protocol; the current baseline avoids introducing image metadata or held-out split priors into normalization.

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

Run lightweight local verification:

```powershell
pytest
```

Train CSI2Pose direct regression only on the Linux server after pulling the latest code:

```bash
python train.py --config configs/csi2pose_tcn.yaml --dataset-root /path/to/mmfi_pose.h5 --decoder regression
```

Train the historical heatmap decoder only on the Linux server:

```bash
python train.py --config configs/csi2pose_tcn.yaml --dataset-root /path/to/mmfi_pose.h5 --decoder heatmap
```

During startup, `train.py` prints window-building progress for each split. It builds train and validation windows before training, then delays test-window construction until the best checkpoint is ready for final evaluation.

Evaluate a saved CSI2Pose checkpoint only on the Linux server, using the same decoder that created the checkpoint:

```bash
python test.py --config configs/csi2pose_tcn.yaml --dataset-root /path/to/mmfi_pose.h5 --decoder regression --checkpoint runs/csi2pose_regression_tcn/best.pt
```

Save GT/prediction pose comparison PNGs during server-side checkpoint evaluation:

```bash
python test.py --config configs/csi2pose_tcn.yaml --dataset-root /path/to/mmfi_pose.h5 --decoder regression --checkpoint runs/csi2pose_regression_tcn/best.pt --visualize
```

Visualization samples one middle frame from a randomly selected sequence window for each `(action, environment, sample)` group in the evaluated split. Keypoints are restored to pixel-coordinate scale before plotting. By default images are written under the checkpoint directory at `visualizations/<split>/`; use `--visualization-dir` and `--visualization-seed` to override the output location or sampling seed on the Linux server.

## Coding Style & Naming Conventions
Use Python 3.10+ syntax, type hints, and `pathlib.Path` for paths. Group imports as standard library, third-party, then local. Follow existing naming: `snake_case` functions and variables, `PascalCase` classes, and uppercase constants such as `SPLIT_NAMES` or `CSI_SHAPE`. Use 4-space indentation. Keep comments focused on dataset assumptions, tensor shapes, alignment constraints, and normalization or cleaning behavior.

## Testing Guidelines
Automated tests use `pytest`. Keep fixtures synthetic and small on the local machine. Prioritize coverage for raw-directory discovery, frame-count validation, frame-name alignment, HDF5 round-tripping, split generation, shape validation, sequence-window construction, and CSI cleaning or normalization edge cases.

Local verification should stay lightweight and code-focused. Do not run training, evaluation, long data-packing jobs, checkpoint generation, metrics export, or output-producing experiment commands on the local machine.

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

Use this repository checkout only for code edits and lightweight code verification. For any future model training, validation, evaluation, visualization, metrics export, checkpoint export, or dataset/output generation, commit and push the code first, then run those commands only on the Linux server after `git pull`.
