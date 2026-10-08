# Checkpoints, independent evaluation and aggregation

DRIVE clean-image evaluation lives in `drive.py`, with shared vessel metrics in
`topology.py`. No parent research tree is required. Checkpoints retain the source
hashes from their training run; changes trigger the source guard. That guard is
not automatically bypassed; see the explicit mismatch option below.

Run these commands from the **complete submission-package root**. No new training
is launched by either evaluation or aggregation. Existing research results are
never overwritten. New output directories must not already exist.

## The three commands

```bash
python -I -B training/run.py --experiment plain_relative --dataset-root datasets/pathfinder --output-dir outputs/plain_relative_run1 --device cuda --execute
python -I -B evaluation/evaluate.py --run-dir outputs/plain_relative_run1 --dataset-root datasets/pathfinder --output-dir outputs/plain_relative_test --selection both --device cuda
python -I -B evaluation/aggregate.py --inputs outputs/plain_relative_test --output-dir outputs/plain_relative_table --expected-seeds 23 7 42
```

Use the appropriate experiment preset and prepared dataset directory. Default
training uses the prescribed epoch budget and three seeds. For a different seed
set, change both the training filter and `--expected-seeds`. Evaluation defaults
to `best_val`; `both` produces separate best/final rows, never their pooled mean.
You can evaluate one file with `--checkpoint .../best_val.pt` instead of `--run-dir`.

The aggregate directory contains:

- `summary.json`: exact per-seed records, full grouping/provenance and aggregates.
- `summary.csv`: long-form metric rows with mean and **sample** SD (`ddof=1`).
- `summary.md`: a readable table. Classification accuracy is displayed as percent;
  machine-readable JSON/CSV keep the original fractions. Other metrics keep native units.

Single-seed SD is `null`/blank/`N/A`, **not zero**, and the row is labelled preliminary.
Duplicate seeds/checkpoints, missing expected seeds, non-finite metrics, incompatible
test units and incomplete evaluation manifests are rejected. Different models,
widths, training splits, targets, protocols, checkpoint selections, code versions
or evaluation devices/precision are placed in separate groups. Group IDs link the
short tables to the full specifications in JSON. No significance test or paper
acceptance claim is made by this script.

## What training now saves

The public `training/run.py` launcher enables export for all seven supervised
runner families, including DRIVE. Each model/split/seed gets:

```text
<job>/checkpoints/s<seed>_<configuration-hash>/
  best_val.pt
  final.pt
  checkpoints.json
```

The checkpoint contains the complete model state, constructor settings, actual
runner arguments, selected epoch, source/metadata hashes and test-evaluation RNG
state. Validation accuracy selects Pathfinder models; validation Dice selects DRIVE.
Strict improvement keeps the first epoch on ties. CPU tensors are cloned, so later
updates cannot overwrite an earlier selection. hGRU now also selects epoch one if
validation accuracy is zero (the old initial score of zero could select no model).

These are **evaluation checkpoints**, not resumable optimiser checkpoints. SPT
reconstruction retains its existing encoder/autoencoder exports; assess downstream
classification with the SPT fine-tuning presets, not with an autoencoder test score.
Direct legacy runner calls do not enable the new exporter automatically; use the
public launcher. Its metadata and output guards are part of the release workflow.

Export adds memory, copying and I/O overhead. New elapsed times are not substitutes
for the historical paper timing measurements. Legacy epoch logs/summary files
remain for provenance; use this aggregation tool for sample-SD test tables.

## Evaluation contracts

Checkpoints are loaded with `weights_only=True` and strict state-key matching.
Model reconstruction requests no ImageNet download. Only known model constructors
and evaluation runners are supported; do not load untrusted checkpoints.

Test metadata/target-version summary must match their training hashes. Prepared
datasets can be relocated, but their CSV/summary contents must be unchanged.
This is **metadata identity**, not proof of original pixel-byte identity: run the
data-generation verification separately. Source changes are rejected by default;
`--allow-source-mismatch` explicitly permits and records a changed-code evaluation.
Such results are grouped separately. No legacy state dict is silently upgraded.
An incomplete training run requires an explicit checkpoint rather than whole-run
evaluation. Failed evaluation directories cannot be silently aggregated.

Pathfinder uses the preserved runner's normalisation, batch size 64 and evaluation
precision, without augmentation. Global-control MAE is named `global_betti0_mae`,
not relative MAE. Label-only Plain has no auxiliary metric. hGRU's initial hidden
state is stochastic even in evaluation: saved RNG state replays the recorded test
pass on the same device type. `--allow-device-change` explicitly permits a seeded
cross-device evaluation, but it is **not exact replay**. Same device type alone
does not guarantee bitwise equality across hardware/library versions.

DRIVE evaluates clean RGB images (no baked query markers), inside the stored FOV,
with the retained FP32 topology evaluator and fixed probability threshold 0.5.
Dice, IoU, clDice, APLS, Betti errors and the retained width/junction metrics are
computed **per image**, then averaged within a seed. Seed SD is computed across
those image means; query pairs are not counted as independent images. Per-image
records are included. These numbers must not be mixed with training-time
query-conditioned/pair-level Dice. Both best and final weights can be evaluated.

Checkpoint metadata omits local input/output paths. Launcher logs/configs can still
contain local paths and must be anonymised before sharing. Hardware/library
versions are recorded in runtime/evaluation reports, without a host or user name.

## Bounded verification

```bash
python -I -B evaluation/check.py --output-dir outputs/evaluation_check_01
```

The check performs a three-seed two-epoch fixture run through all three public
commands; re-evaluates both checkpoint selections for every Pathfinder runner
family and all four DRIVE conditions; tests hGRU RNG replay, strict model loading,
snapshot isolation, sample SD, and invalid-input rejection. DRIVE's actual topology
evaluator runs on small synthetic images (not a mock). The pretrained runner uses
scratch weights in this offline check. hGRU's full-resolution check is forward-only;
short BPTT is covered by `training/check.py`. Check artefacts include many weights
and can take substantial temporary disk space; they are not release contents.

This completes the new-run code path, **not historical result reproduction**.
Full GPU training, the Plain inter-runner protocol discrepancy, historical DRIVE
checkpoint/count provenance and paper-number reconciliation remain separate checks.
