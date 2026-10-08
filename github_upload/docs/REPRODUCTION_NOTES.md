# Reproduction notes

This package provides data preparation, training, checkpoint evaluation, and
seed aggregation for the experiments described in the paper. The notes below
describe the supported workflows and reproduction scope.

- **Plain protocols:** controls/dense/SPT fine-tuning use CUDA FP16 and grouped
  weight decay; the relative data-scaling preset uses BF16 and decay on all CNN
  parameters. Runner schedule/batch handling is also retained. Historical
  source/configuration matching remains to be resolved before describing these
  as exactly matched controls. Remainder-batch handling does not change a 4K or
  16K split with batch size 32 because those sizes are divisible by 32.
- **Targets:** standard Pathfinder uses image-derived analytic V2 targets.
  Dense-clutter retains its recorded antialiased/pre-quantisation construction;
  it is not silently migrated to V2. DRIVE targets are derived from manual vessel
  annotations, not image-only self-supervision.
- **Data identity:** the 32K extension has frozen scene metadata, but original
  pixel hashes are unavailable for its added 16K scenes. Complete synthetic
  regeneration/original-pixel comparison remains open. DRIVE FOV masks are
  RGB-derived; the annotations themselves are not modified.
- **DRIVE training:** the implemented loss is `0.5 * BCE_FOV + 0.5 * Dice_FOV`,
  plus the selected auxiliary loss. Current U-Nets retain historical shared
  branches, including an unused
  65-parameter RoI head; parameter counts/checkpoint provenance differ from some
  older records. Strict checkpoint loading is required.
- **Release fixes:** new DRIVE runs seed Python random and clone CPU best-state
  tensors. SPT transfer uses strict, weight-only encoder loading. The hGRU
  best-score initialisation permits selecting a zero-accuracy first epoch.
  These fixes do not retroactively establish the exact historical execution.
- **Metrics:** aggregate with the supplied sample-SD evaluator, not legacy
  runner population-SD summaries. Capacity-Swin's legacy `first_90_epoch` field
  uses test accuracy; it is not a validation threshold. Its best-checkpoint
  selection remains validation-based.
- **Coverage:** exact historical table/figure regeneration and image-level DRIVE
  significance tests are not yet integrated. Historical per-seed/per-image
  records, pretrained result checkpoints and qualitative predictions are not
  bundled. Their absence is not resolved by a successful software smoke test.
- **Environment and timing:** dependencies are ranges, not the recovered training
  environment lock. CPU fixture checks do not certify GPU/accuracy equivalence
  or a clean install. Checkpoint export/setup/evaluation can add runtime; new
  timings should not replace the paper's recorded cost measurements.
- **Distribution:** the third-party hGRU Gabor initialisation is not redistributed.
  The hGRU baseline requires the separately obtained, checksum-verified upstream
  file documented in the [hGRU asset setup](../models/assets/HGRU_ASSET.md).

For offline checks, use `models/check.py --backward`, the generation/loader unit
tests, or the documented training/evaluation fixture checks. Run these with fresh
scratch directories; they produce test artefacts, not paper results.
