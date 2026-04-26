# Repository Guidelines

## Project Structure & Module Organization
- `dataloader.py`: Core module for discovering raw MM-Fi samples, validating frame alignment, building split indices, cleaning CSI features, packing one HDF5 dataset, loading HDF5 splits, creating PyTorch `DataLoader` instances, and previewing split contents.
- `scripts/build_h5_dataset.py`: Command-line wrapper that packs the raw MM-Fi directory tree into one `.h5` or `.hdf5` dataset file.

This repository is a small baseline subproject focused on dataset preparation and inspection only. It does not currently include model, training, evaluation, or test modules.

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

## Coding Style & Naming Conventions
Use Python 3.10+ syntax, type hints, and `pathlib.Path` for paths. Group imports as standard library, third-party, then local. Follow existing naming: `snake_case` functions and variables, `PascalCase` classes, and uppercase constants such as `SPLIT_NAMES` or `CSI_SHAPE`. Use 4-space indentation. Keep comments focused on dataset assumptions, tensor shapes, alignment constraints, and normalization or cleaning behavior.

## Testing Guidelines
Automated tests are not present yet in this subproject. When adding tests, use `pytest`, name files `test_*.py`, and keep fixtures synthetic and small. Prioritize coverage for raw-directory discovery, frame-count validation, frame-name alignment, HDF5 round-tripping, split generation, shape validation, and CSI cleaning or normalization edge cases.

## Commit & Pull Request Guidelines
Use concise imperative commits, for example `Update AGENTS for baseline dataset project`. Pull requests should include a summary, commands run, dataset assumptions, and any relevant tensor-shape or frame-count output. Do not commit generated datasets, virtual environments, or machine-specific paths.

## Security & Configuration Tips
Do not hard-code private dataset locations beyond documented defaults. Pass dataset paths with `--dataset-root` and keep large or sensitive data outside version control.

## Agent-Specific Instructions
Write repository-facing agent notes, documentation, and code comments in English. Keep comments aligned with surrounding style. Use Chinese for conversational replies unless the user requests another language.

Whenever project changes affect commands, structure, conventions, testing, configuration, or agent workflow, update this `AGENTS.md` file in the same turn.

After each project modification, commit the change and push it to the configured GitHub remote in the same turn unless the user explicitly asks not to push.

Before changing code, apply the `karpathy-guidelines` skill: state assumptions when needed, prefer the smallest working change, avoid unrelated refactors, and verify the result with a concrete check.

Before running project code or tests, activate the existing Conda environment with `conda activate WiFiPose` to ensure commands run in the established project environment.
