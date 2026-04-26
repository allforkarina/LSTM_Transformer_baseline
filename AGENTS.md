# Repository Guidelines

## Project Structure & Module Organization
- `dataloader.py`: Core module for discovering samples, packing HDF5 files, loading splits, creating PyTorch `DataLoader` instances, and previewing split contents.
- One packed HDF5 can now hold both `action_env` and `frame_random` split schemes; training and evaluation default to `action_env` and can switch with `--split-scheme`.
- `models/`: PyTorch model code, including the full WiFlow model, a structured CSI token encoder, a joint-query cross-attention decoder, asymmetric CNN, and axial attention stages.
- `train.py`: Root-level training entrypoint for WiFlow pose regression, including losses, metrics, optimizer, scheduler, checkpointing, and CSV logging.
- `eval.py`: Root-level evaluation entrypoint for loading checkpoints, computing test metrics, and saving CSI/skeleton visualizations.
- `scripts/build_h5_dataset.py`: Command-line wrapper that builds a single `.h5`/`.hdf5` dataset from the raw MM-Fi directory structure.
- `tests/`: `pytest` unit tests. Mirror module names such as `tests/test_dataloader.py`, `tests/test_wiflow_model.py`, or `tests/test_wiflow_decoder.py`.
- `.gitignore`: Excludes Python caches, local environments, generated datasets, checkpoints, and editor files from Git.

Generated datasets can be large and should not be committed. Keep raw dataset roots outside the repository.

## Build, Test, and Development Commands
Use the existing Conda environment for development commands:

```powershell
conda activate WiFiPose
pip install numpy scipy h5py tqdm torch pytest
```

Build an HDF5 dataset:

```powershell
python scripts\build_h5_dataset.py --dataset-root D:\path\to\raw\dataset --output-path data\mmfi_pose.h5 --seed 42
```

Inspect an HDF5 dataset:

```powershell
python dataloader.py --dataset-root data\mmfi_pose.h5 --preview
```

Run tests:

```powershell
pytest
```

Run a quick training sanity check:

```powershell
python train.py --dataset-root data\mmfi_pose.h5 --epochs 5 --subset-size 32 --output-dir outputs\sanity
```

Run the default training configuration:

```powershell
python train.py --dataset-root data\mmfi_pose.h5 --epochs 50 --batch-size 64 --output-dir outputs\train
```

The default training configuration uses `OneCycleLR`, gradient clipping, a staged bone-loss curriculum, and stronger AdamW weight decay than the original baseline. The model now keeps 29 spatial CSI tokens before the joint-query decoder instead of collapsing them to 17 slots inside the encoder.

Run the frame-random split configuration:

```powershell
python train.py --dataset-root data\mmfi_pose.h5 --split-scheme frame_random --epochs 50 --batch-size 64 --output-dir outputs\train_frame_random
```

Evaluate one checkpoint:

```powershell
python eval.py --dataset-root data\mmfi_pose.h5 --checkpoint outputs\train\best_val_mpjpe.pth --output-dir outputs\eval
```

Evaluate against the frame-random split:

```powershell
python eval.py --dataset-root data\mmfi_pose.h5 --checkpoint outputs\train_frame_random\best_val_mpjpe.pth --split-scheme frame_random --output-dir outputs\eval_frame_random
```

## Coding Style & Naming Conventions
Use Python 3.10+ syntax, type hints, and `pathlib.Path` for paths. Group imports as standard library, third-party, then local. Follow existing naming: `snake_case` functions/variables, `PascalCase` classes, and uppercase constants such as `SPLIT_NAMES`. Use 4-space indentation. Keep comments focused on dataset assumptions, shapes, and normalization.

## Testing Guidelines
Automated tests use `pytest`. Add tests for split generation, path validation, shape validation, normalization edge cases, model shape contracts, and HDF5 round-tripping. Name files `test_*.py` and tests `test_<behavior>()`. Use temporary directories and tiny synthetic fixtures.

Training and evaluation outputs are written under `outputs/` by default. Checkpoints include `best_val_mpjpe.pth`, `best_val_pck_0_2.pth`, and `last.pth`; epoch metrics are appended to `train_log.csv`. Evaluation visualizations are saved as `.png` files grouped by action/environment samples.

## Commit & Pull Request Guidelines
This checkout has no `.git` history, so no convention can be inferred. Use concise imperative commits, for example `Add HDF5 split preview`. Pull requests should include a summary, commands run, dataset assumptions, and relevant shape or frame-count output. Do not commit generated datasets, virtual environments, or machine-specific paths.

## Security & Configuration Tips
Do not hard-code private dataset locations beyond documented defaults. Pass dataset paths with `--dataset-root` and keep large or sensitive data outside version control.

## Agent-Specific Instructions
Write repository-facing agent notes, documentation, and code comments in English. Keep comments neatly aligned with surrounding style. Use Chinese for conversational replies unless the user requests another language.

Whenever project code changes, update this `AGENTS.md` file in the same turn if the change affects commands, structure, conventions, testing, configuration, or agent workflow.

After each project modification, commit the change and push it to the configured GitHub remote in the same turn unless the user explicitly asks not to push.

Before changing code, apply the `karpathy-guidelines` skill: state assumptions when needed, prefer the smallest working change, avoid unrelated refactors, and verify the result with a concrete check.

Before running project code or tests, activate the existing Conda environment with `conda activate WiFiPose` to ensure commands run in the established project environment.
