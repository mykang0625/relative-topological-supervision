# Load the prepared datasets

This folder can be copied independently. It needs NumPy, Pillow and PyTorch,
but no generator, model, torchvision, pandas, SciPy or GPU. Data must already
have been prepared; nothing is downloaded, generated or written by a loader.
Run examples from the directory containing `data_loading/`.

There are two implementations: `pathfinder.py` (standard and dense-clutter)
and `drive.py`. `test_loading.py` tests small fixtures, and `check.py` reads
real prepared datasets. The requirements describe compatible APIs, not a
recovered historical training environment.

## Pathfinder and dense-clutter

```python
from data_loading.pathfinder import load_split_data, normalise_images, apply_augmentation
import torch

X, Y, D, B, query_yx = load_split_data("datasets/pathfinder", "train_4000")
batch = normalise_images(X[:32])
generator = torch.Generator().manual_seed(23)
batch = apply_augmentation(batch, "dihedral_shift", generator)
```

- `X`: CPU uint8 `[N,128,128]`; `Y`: int64 `[N]` connectivity labels.
- `D`: float32 `[N]` stored `delta_betti0_ssl` relative targets.
- `B`: float32 `[N]` stored `betti0_initial_ssl`, the **dashed-raster** global target.
- `query_yx`: NumPy float32 `[N,2,2]` native pixel coordinates for diagnostics;
  they are not an additional input to the classification model.
- Normalisation is exactly `X.float().unsqueeze(1) / 127.5 - 1.0`.
- Training augmentation draws one transform per batch, as in the original
  runners. Validation and test use normalisation only. Do not move this transform
  into a per-sample Dataset or change the runner's RNG/shuffle order.
- The input is only the existing marker-visible dashed raster. No bridge,
  auxiliary target or separate query channel is rendered into the input.

Use `train_4000`, `train_16000` or `train_32000` explicitly, after generating the
corresponding stage. Fixed smaller selections are also supported. Missing CSVs
or images cause an error, never automatic resampling or a different split.
The prepared standard `train.csv` remains the 4K selection after expansion.

For dense-clutter, change the root to `datasets/dense_clutter`. Its separately
recorded legacy targets remain legacy; the loader accepts that declared version
without converting it to V2. Standard data must declare analytic V2. Missing,
non-finite or inconsistent targets cause an error; there is no online computation
or zero-target fallback. Only `dashed_with_points` is supported: paired-view
metadata contains dashed SSL targets even when another raster path is selected.

Compatibility interfaces preserve the original paper runners:

- `load_scaling_split(...)` returns four tensors. Its fourth entry remains the
  scene-level `betti0` used by those runners (unused by their relative loss),
  **not** the raster-level global control above.
- `load_label_split(...)` returns only images and labels for label-only hGRU.
  Auxiliary targets are neither required nor returned.
- `load_raw_images(...)` returns only images for masked-reconstruction SPT.

The original runners still own batch shuffling and training-only target
permutation. Keep their RNG implementations: the historical NumPy and PyTorch
permutations are not interchangeable merely because the seed number is equal.
Stored CSVs and validation/test targets must never be permuted.

## DRIVE

```python
import random
import numpy as np
import torch
from data_loading.drive import DRIVEPairDataSpec, build_drive_pair_dataloaders

random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
spec = DRIVEPairDataSpec(dataset_root="datasets/DRIVE", num_workers=0)
loaders = build_drive_pair_dataloaders(spec)
batch = next(iter(loaders["train"]))
```

Use a `if __name__ == "__main__":` guard when using multiprocessing on Windows.
The builder keeps the original global-RNG behaviour. Python `random` controls
the per-item augmentation when `num_workers=0`; seed it as well as NumPy and
PyTorch in calling code. Fix worker count when comparing repeated executions.

Images are resized to 512x512 (RGB bilinear, vessel/FOV masks nearest-neighbour).
Query coordinates are scaled with `(size-1)/(native_size-1)`. Training applies
the same D4/zero-filled translation to RGB, both masks and query coordinates,
then bakes the query discs. Validation/test disable augmentation. RGB uses
ImageNet normalisation by default. Native-resolution stored targets are kept;
they are not recomputed on resized or translated masks.

Each batch includes `image [B,3,512,512]`, `vessel_mask [B,1,512,512]`,
`fov_mask [B,1,512,512]`, `coordinates_yx [B,2,2]`, `coordinates_xyxy [B,4]`,
labels/global/relative targets `[B]`, and `pair_id`/`image_id` lists. Coordinates
are normalised to [0,1]. Signed `delta_betti1` is preserved. All masks and targets
are read from the prepared data: the loader does not create human annotations.

The split is 16/4/20 images and 800/200/1,000 queries. The builder rejects
image overlap between splits. Query pairs from one image are not independent
test samples. These are pair-training/evaluation batches; final clean-image
segmentation evaluation uses the separate evaluation script, without query discs.

## Check

```bash
python -I -B data_loading/test_loading.py
python -I -B data_loading/check.py --pathfinder-root datasets/pathfinder
python -I -B data_loading/check.py --pathfinder-root datasets/pathfinder --train-split train_16000 --all-items
python -I -B data_loading/check.py --dense-root datasets/dense_clutter
python -I -B data_loading/check.py --drive-root datasets/DRIVE --all-items
```

The default check validates all selected metadata rows and file existence,
checks split separation, and decodes bounded Pathfinder samples / DRIVE batches.
`--all-items` decodes all selected Pathfinder images and visits every DRIVE pair.
It does not verify frozen checksums, recompute topology, train a network, or
establish reproduction of paper accuracy. Use the generation verifier for
reference metadata/pixel equality. No GPU is used by these checks.
