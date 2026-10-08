# Relative topological supervision

Implementation of relative topological supervision for two-dimensional
Pathfinder connectivity and DRIVE retinal vessel segmentation experiments.

## Setup

Use Python 3.10+ with compatible PyTorch/torchvision builds, and run commands from
this folder. The dependency ranges are not a lock of the historical environment.

```bash
python -m pip install -r requirements.txt
python -I -B tools/verify_package.py --help-checks
```

The hGRU baseline requires a separately obtained Gabor initialisation file; see
[`models/assets/HGRU_ASSET.md`](models/assets/HGRU_ASSET.md).

## Reproduce an experiment

1. Prepare the data using [data_generation/README.md](data_generation/README.md).
   It covers Pathfinder 4K → 16K → 32K, dense-clutter and DRIVE. DRIVE source
   images/manual annotations must be supplied locally; they are not bundled.
2. Select an experiment in [the experiment guide](docs/EXPERIMENTS.md).
   The executable settings are in `configs/experiments.yaml`.
3. Train, evaluate and aggregate. For example, after preparing `datasets/pathfinder`:

```bash
python -I -B training/run.py --experiment plain_relative --dataset-root datasets/pathfinder --output-dir outputs/plain_relative --dry-run
python -I -B training/run.py --experiment plain_relative --dataset-root datasets/pathfinder --output-dir outputs/plain_relative --device cuda --execute
python -I -B evaluation/evaluate.py --run-dir outputs/plain_relative --dataset-root datasets/pathfinder --output-dir outputs/plain_relative_test --selection both --device cuda
python -I -B evaluation/aggregate.py --inputs outputs/plain_relative_test --output-dir outputs/plain_relative_table --expected-seeds 23 7 42
```

Output directories must be new. Training defaults to a no-write dry-run unless
`--execute` is supplied. ImageNet experiments require cached weights or explicit
`--allow-weight-downloads`. Evaluation does not train or download weights.

## Code map

| Folder | Purpose |
| --- | --- |
| [data_generation](data_generation/README.md) | Data preparation and auxiliary targets; frozen manifests and rendering vendor code |
| [data_loading](data_loading/README.md) | Prepared-data loaders and augmentation |
| [models](models/README.md) | Backbones, DRIVE U-Nets, SPT and hGRU asset setup |
| [training](training/README.md) | Public launcher, checkpoint export and experiment implementations |
| `configs/` | Executable experiment presets |
| [evaluation](evaluation/README.md) | Held-out checkpoint evaluation, DRIVE metrics and seed aggregation |
| `tools/` | Package integrity/import/CLI check and file hashes |
| `docs/` | Experiment guide and reproduction limitations |

Keep generated data in `datasets/` and outputs in `outputs/` (both Git-ignored).
Do not distribute personal paths or unfiltered runtime logs. This bundle does not
include datasets, trained checkpoints, private audit reports or historical result
archives. Third-party dependencies and separately obtained assets are documented
with the relevant components.
