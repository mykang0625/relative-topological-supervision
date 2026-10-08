# Experiment guide

Use `training/run.py --list` to list presets. All settings are in
`configs/experiments.yaml`; runners keep their prescribed protocols.

| Study | Preset(s) | Data |
| --- | --- | --- |
| Plain supervision controls | `plain_label_only`, `plain_global`, `plain_permuted`, `plain_relative` | Pathfinder, 4K |
| Training-set size | `data_scaling` | Nested Pathfinder subsets, up to 32K |
| Model capacity | `capacity_cnn`, `capacity_swin` | Pathfinder, 16K |
| Label-only hGRU | `hgru` | Nested Pathfinder subsets |
| ImageNet initialisation | `pretrained`, `pretrained_label_only` | Pathfinder, 4K |
| Self-pretraining | `spt_pretrain` | Pathfinder training images, 4K |
| SPT downstream training | `spt_label_only`, `spt_relative` | Pathfinder, 4K; supply `--spt-dir` |
| Dense-clutter stress test | `dense_label_only`, `dense_relative` | Separate dense-clutter dataset |
| Vessel segmentation | `drive` | DRIVE, image-disjoint 16/4/20 train/validation/test |

The `hgru` preset requires the separately obtained upstream Gabor file described
in the [hGRU asset setup](../models/assets/HGRU_ASSET.md).

Default seeds are 23, 7 and 42. Data-scale presets accept `--splits train_16000`;
multi-model presets accept `--models`. DRIVE additionally accepts
`--conditions seg_only global_betti shuffled_control qati_only`. For example:

```bash
python -I -B training/run.py --experiment drive --dataset-root datasets/DRIVE --output-dir outputs/drive_plain --models plain --conditions seg_only qati_only --dry-run
```

After SPT pretraining, pass the directory containing the encoder files to the
downstream preset via `--spt-dir`. See [training details](../training/README.md)
for filters, transfer loading, optimiser/precision differences and saved files.
Do not interpret all presets as having an identical optimisation protocol.

## Evaluation

Use `evaluation/evaluate.py` with `--selection best_val`, `final`, or `both`.
Selection uses validation accuracy for Pathfinder and validation Dice for DRIVE;
held-out test scores do not select the checkpoint. Use `evaluation/aggregate.py`
to compute seed means and sample standard deviations. Best/final selections and
different protocols remain separate. Single-seed SD is unavailable, not zero.

DRIVE evaluation is on clean images, with per-image results averaged within each
seed. Query pairs are not independent test images. See the
[evaluation guide](../evaluation/README.md) for checkpoint/source guards.

Paper-specific significance tests and historical figures are not part of this
workflow yet; see [reproduction notes](REPRODUCTION_NOTES.md).
