# Reproduce the datasets

This folder contains all code and reference metadata needed to prepare the
paper's **Pathfinder**, **dense-clutter**, and **DRIVE** datasets. It can be copied
to a new repository without `iclr2027/`, `src/`, model code, PyTorch or a GPU.
DRIVE images and manual vessel annotations must be obtained separately.
Run the commands below from the directory **containing** `data_generation/`.

## Layout

```text
data_generation/
├── README.md                       # start here
├── requirements.txt                # one preparation environment
├── prepare_4k.py                    # public command 1
├── extend_to_16k.py                 # public command 2
├── extend_to_32k.py                 # public command 3
├── prepare_dense_clutter.py         # public command 4
├── prepare_drive.py                 # public command 5: existing manual labels
├── build_drive_targets.py           # public command 6: derived query targets
├── _lib/                           # eight private implementation modules
├── vendor/                         # snakes.py, snakes2.py and MIT licence
├── manifests/                      # fixed scene/split/query records + SHA-256
├── test_generation.py              # safety/regression tests
└── check.py                        # isolated-folder and real-data checks
```

The six preparation scripts are the public interface. Other Python modules are
implementation details, not additional dataset variants. The compressed
manifests are required inputs, not generated caches. Keep this folder intact,
including vendor attribution and licence files.

## Install

Python 3.10 or later is required. The pinned preparation versions were checked
with Windows Python 3.10.20; they are not the historical training lockfile.

```bash
python -m pip install -r data_generation/requirements.txt
```

No PyTorch, GPU or scikit-image is needed for preparation.

## Generate

Use new output directories. The commands never modify the source research corpus.

```bash
# Standard Pathfinder: fixed held-out scenes and nested training sets.
python data_generation/prepare_4k.py --output-root datasets/pathfinder --workers 4
python data_generation/extend_to_16k.py --output-root datasets/pathfinder --workers 4
python data_generation/extend_to_32k.py --output-root datasets/pathfinder --workers 4

# Independent dense-clutter stress test: 16K train + 900 val + 900 test.
python data_generation/prepare_dense_clutter.py --output-root datasets/dense_clutter --workers 4

# DRIVE: locally obtained RGB images + manual annotations, then auxiliary targets.
python data_generation/prepare_drive.py --source-root /path/to/obtained/DRIVE --output-root datasets/DRIVE
python data_generation/build_drive_targets.py --dataset-root datasets/DRIVE --workers 2
```

Add `--dry-run` to inspect the plan without writing, `--resume` after interruption,
or `--verify-only` to check a completed output. Standard Pathfinder verification
uses the largest completed stage. Smaller training sets are fixed CSV selections,
not separately resampled or copied images. DRIVE uses image-disjoint 16/4/20 splits
and 50 fixed query pairs per image; it does not generate vessel annotations.

The first stage creates 4,000 training scenes plus 900 validation and 900 test
scenes. Expansion adds 12,000 then 16,000 training scenes and preserves held-out
data. The available nested training selections are 250/500/1K/2K/4K/8K/16K/32K.
`metadata/train.csv` remains the 4K selection; scaling uses the size-specific CSVs.
Each scene has four paired views; training uses `dashed_with_points` by default.

DRIVE input must contain `training/images/21_training.tif` through `40_training.tif`,
`training/1st_manual/21_manual1.gif` through `40_manual1.gif`, and corresponding
`test/images/01_test.tif` through `20_test.tif` with `test/1st_manual/01_manual1.gif`
through `20_manual1.gif`. PNG versions with these same stems are also accepted.
Exactly one recognised file per stem is required. Images are not resized during
preparation. Outputs use `metadata_qati_pairs_v1/`, as expected by the training loader.

Existing files with different pixels/metadata cause an error, not an overwrite.
An interrupted build requires `--resume`; a missing file in a completed build is
reported as corruption. An unmanaged nonempty output directory is rejected.

## Verify before a long build

```bash
python -I -B data_generation/test_generation.py
python -I -B data_generation/check.py --output-root scratch/check --with-smoke
```

The second command copies only this folder to scratch, runs six CLI help checks,
unit tests, a dry-run, and bounded real generation (14 standard and eight dense
scenes). Add `--drive-source /path/to/obtained/DRIVE` to also prepare all 40 DRIVE
images and independently recompute/verify all 2,000 query records. No download,
training or full synthetic-corpus generation is performed by this check.

## Reproducibility boundaries

- Standard Pathfinder uses the recorded SciPy dilation semantics and analytic
  V2 targets (capsule radius 1.5 px, foreground threshold 0.1, 8-connectivity).
  Original pixel references cover 17,800 scenes; the extra 16K has
  reference metadata but not original pixel hashes. Do not claim original-pixel
  equality for that extension.
- Dense-clutter uses OpenCV and its recorded legacy antialiased/pre-quantisation
  targets. Do not substitute V2 or present these target conventions as identical.
- DRIVE targets come from manual vessel annotations. FOV masks are reconstructed
  from RGB, not taken from the official mask files. Query coordinates are frozen;
  all label/topology values are recomputed and checked. The capsule radius is 6
  native pixels, with foreground-8/background-4 connectivity. FOV construction:
  any RGB channel >10, fill holes, retain the largest 4-connected component.
- Small checks do not establish full synthetic regeneration, a clean installation
  on another platform, or reproduction of trained-model accuracy.

## What to commit

Commit this complete folder: source, reference manifests, dependency files,
instructions, tests and the upstream licence. Do **not** commit downloaded DRIVE
rasters, generated images, scratch directories, caches or local environment files.
The full submission repository provides a root `.gitignore` for these outputs.
If publishing this folder separately, copy those ignore rules to that repository.
The frozen manifests are sufficient for reproduction; no legacy dataset CLI or
manifest-extraction tool is needed. Do not rebuild these references just to accept
changed output. `vendor/LICENSE` retains the original Pathfinder generator's MIT
licence and copyright; this is not a licence grant for DRIVE or the whole project.

This organisation prepares source for release; it does not itself approve a
public release, complete the project's licensing review, or satisfy anonymous
submission requirements. No GitHub upload is performed by these commands.
