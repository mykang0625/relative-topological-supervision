# Training and experiment settings

Use **one public launcher**, `training/run.py`, and **one executable preset file**,
`configs/experiments.yaml`. Keep this package together: training imports the sibling
`models/`, `data_loading/` and `evaluation/` (for DRIVE topology metrics).
Copying only `training/` is
not sufficient. Install the package-root `requirements.txt` with a matching
PyTorch/torchvision build first. No historical environment lock is claimed.

The eight implementations under `_runners/` preserve the original experiment
families and filenames. Use the public launcher instead of former research-tree
paths. The single `configs/experiments.yaml` is consumed by the launcher;
redundant historical YAMLs and compatibility commands are no longer shipped.

## Start safely

Run from the submission-package root. Dataset paths are the output directories
you supplied to the preparation commands, not automatically inferred locations.

```bash
python -I -B training/run.py --list
python -I -B training/run.py --experiment plain_relative --dataset-root datasets/pathfinder --output-dir outputs/plain_relative --dry-run
```

The default is a **no-write dry-run**: it prints the resolved settings and argument
arrays without importing models, downloading weights, loading images or training.
It does not establish that the input data exist. To train, replace `--dry-run` with
`--execute`. An existing output directory is rejected; use a fresh name. There is
no implicit resume/overwrite. Direct historical CLIs do not provide these guards.

```bash
python -I -B training/run.py --experiment plain_relative --dataset-root datasets/pathfinder --output-dir outputs/plain_relative_run1 --device cuda --execute
```

ImageNet experiments use the local torchvision cache. Downloads are blocked unless
you explicitly add `--allow-weight-downloads`. `--device cpu` disables CUDA for
the child process; `--device cuda` fails if CUDA is unavailable. Full experiments
are expensive on CPU. `--epochs` is an explicitly recorded nonstandard override,
not a reproduction of the fixed-budget result.

## Experiment map

| Preset | Experiment | Default split / seeds |
| --- | --- | --- |
| `plain_label_only`, `plain_global`, `plain_permuted` | Plain supervision controls | 4K / 23,7,42 |
| `plain_relative` | Recorded relative Plain row (data-scaling implementation) | 4K / 23,7,42 |
| `data_scaling` | Plain, residual and scratch Swin | Nested 250–32K / 23,7,42 |
| `capacity_cnn` | Ten widths each for Plain and residual | 16K / 23,7,42 |
| `capacity_swin` | Six scratch Swin configurations | 16K / 23,7,42 |
| `hgru` | Label-only recurrent reference | Nested 250–32K / 23,7,42 |
| `pretrained`, `pretrained_label_only` | ImageNet ResNet-18/Swin-T, relative or label-only loss | 4K / 23,7,42 |
| `spt_pretrain` | Masked reconstruction on training images only | 4K / 23,7,42 |
| `spt_label_only`, `spt_relative` | SPT encoder transfer and downstream fine-tuning | 4K / 23,7,42 |
| `dense_label_only`, `dense_relative` | Separate dense-clutter corpus | Full 16K `train` / 23,7,42 |
| `drive` | Four backbones × four conditions | 16/4/20 images, 800/200/1,000 queries / 23,7,42 |

Before running `hgru`, obtain the checksum-verified upstream Gabor file as
described in [`../models/assets/HGRU_ASSET.md`](../models/assets/HGRU_ASSET.md).
The package does not download or substitute this file automatically.

Filter data scaling/hGRU with `--splits train_16000`; filter multi-model presets
with `--models plain` (or another listed model). DRIVE accepts `--models plain`
and `--conditions seg_only qati_only`. `--seeds 23` executes just one seed and must
not be presented as the three-seed result. The 32K preset requires the completed
32K extension; metadata availability is not original-extension pixel verification.

For example, inspect one DRIVE comparison without running it:

```bash
python -I -B training/run.py --experiment drive --dataset-root datasets/DRIVE --output-dir outputs/drive_plain --models plain --conditions seg_only qati_only --dry-run
```

After `spt_pretrain`, pass the directory containing the encoder files, normally
`<output-dir>/spt_pretrain`, as `--spt-dir` to either SPT fine-tuning preset.
The expected files are `plain_w32_spt_s23.pt`, `plain_w32_spt_s7.pt` and
`plain_w32_spt_s42.pt`. Encoder transfer now uses safe weight-only loading and
strict key matching; missing/incompatible weights cannot silently become a
scratch run. Pretraining and fine-tuning each use their own 200-epoch budget.

## Protocol distinctions that must remain visible

All Pathfinder classification conditions use CE; an enabled topological target
adds `0.5 * SmoothL1(beta=1)`. The global Plain condition supplies stored raster
Betti-0 to the scalar auxiliary output used by its original runner. Permuted
controls shuffle **training targets only**, once; validation/test targets and
the on-disk metadata remain unchanged. hGRU has no auxiliary target or loss.

| Implementation family | Optimisation and execution details |
| --- | --- |
| Plain controls, dense-clutter, SPT fine-tuning | AdamW 1e-3, decay 1e-4 only on eligible weight tensors; 5-epoch warmup/cosine to 1e-6; CUDA FP16 autocast + GradScaler |
| CNN data/capacity scaling | AdamW 1e-3, decay 1e-4 on all parameters; 5-epoch warmup/cosine to 1e-6; CUDA BF16 |
| Scratch Swin | AdamW 1e-4, decay 0.05 excluding 1D/norm/bias/relative-position-bias tensors; 10-epoch warmup; CUDA BF16 |
| ImageNet backbones | AdamW 1e-4, decay 1e-4 on all parameters; 5-epoch warmup; CUDA BF16 |
| hGRU | NAdam 1e-3, decay 0, no scheduler, gradient clipping 1, CUDA BF16; one process at a time in the preset |
| SPT reconstruction | FP32, AdamW 1e-3, decay 1e-4, 5-epoch warmup/cosine; no gradient clipping in the recorded runner |
| DRIVE | AdamW 1e-3, decay 1e-4, 60 epochs, batch 16, 5-epoch warmup then cosine; CUDA BF16, clipping 1 |

CPU execution is FP32. Pathfinder classification uses 200 epochs and batch 32.
The original scheduler formulas, first-epoch rates, remainder-batch handling and
augmentation RNG consumption differ across runner families and are retained.
For example, scaling runners drop the final incomplete training batch while the
Plain-control runner includes it. Do not merge these into a single 'matched'
protocol without reconciling the historical sources and run records.

DRIVE's actual segmentation loss is `0.5 * BCE_FOV + 0.5 * Dice_FOV`, with no
connectivity loss. Global or relative auxiliary loss has weight 0.05; seg-only
uses zero. The original DRIVE scheduler decays towards zero, not an explicitly
specified 1e-6 floor. New release runs seed Python `random` as well as NumPy and
PyTorch, fixing the zero-worker augmentation seed gap. CPU best-state snapshots
are cloned so later updates cannot mutate the selected checkpoint. These changes
do not retroactively establish determinism of historical runs.

## Saved outputs and evaluation boundaries

Each invocation saves `run_manifest.json`; each job saves `resolved.json`,
`runtime.json` (actual device/hardware/library versions), `console.log`, and its
original per-seed results. DRIVE also saves its executable `resolved.yaml` and
validation-selected weights. SPT saves encoder and autoencoder state dictionaries.
The public launcher now additionally exports **validation-selected and final
evaluation checkpoints** for every supervised runner. See
[evaluation and aggregation](../evaluation/README.md) for independent test
evaluation and sample-SD tables. Optimiser-state resume is not provided.

Pathfinder selection uses validation accuracy; DRIVE uses validation Dice. The
preserved Pathfinder runners also measure test metrics during training, but never
use test accuracy for checkpoint selection. One legacy reporting exception:
capacity-Swin's `first_90_epoch` uses **test** accuracy, unlike validation-based
threshold fields elsewhere. Do not pool or relabel it as a validation threshold.
Some runner summaries still use population SD: paper aggregation must be rebuilt
from per-seed records with sample SD in the evaluation/release stage.

Runner timing boundaries differ (e.g. hGRU includes data/model setup, DRIVE includes
final topology evaluation). `launcher_elapsed_seconds` additionally includes
process/setup overhead. Do not substitute these measurements for controlled
hardware timing or the historical cost table. Local paths appear in generated
runtime logs/configs; anonymise outputs before sharing them.

## Short checks, not a paper rerun

```bash
python -I -B training/check.py --output-dir outputs/training_check_01
```

Uses a fresh directory with synthetic fixtures, 18 two-epoch CPU cases and no
downloads. It checks training losses, target non-mutation, SPT transfer rejection,
DRIVE checkpoint writing, preset/argument mapping and output guards. The pretrained
runner uses scratch weights in this portable smoke; DRIVE's final topology
evaluation is mocked, not validated by this check; `evaluation/check.py` exercises
the actual topology evaluator. See [reproduction notes](../docs/REPRODUCTION_NOTES.md)
for historical source/environment, DRIVE model-count and paper-result limitations.
