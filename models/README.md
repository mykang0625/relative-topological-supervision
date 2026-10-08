# Models

Copy this entire `models/` folder, including `assets/`, to use the model
definitions without the research tree, datasets or training scripts. Python
3.10+ and compatible PyTorch/torchvision builds are required; see `requirements.txt`.
Importing a module does not create a model or download weights.

## Files and interfaces

| File | Public constructors | Output |
| --- | --- | --- |
| `plain.py` | `PlainNet`, `MultiTaskPlain` | Connectivity logits; multi-task also returns global and relative predictions |
| `cnn_s.py` | `CNNSmall`, `MultiTaskCNNSmall` | Same interface as Plain; residual stack |
| `resnet.py` | `ResNet`, `MultiTaskResNet` | Connectivity logits; multi-task adds one selected auxiliary prediction |
| `swin.py` | `SwinT`, `MultiTaskSwinT`, `SwinCapacityMultiTask`, `SWIN_CONFIGS` | Connectivity logits; multi-task adds one auxiliary prediction |
| `hgru.py` | `HGRU` | Label-only connectivity logits for the paper baseline |
| `drive_unet.py` | `MultiTaskUNet` (`RETAMultiTaskUNet` alias) | Dictionary of segmentation logits and auxiliary predictions |
| `spt.py` | `PlainUNetAutoencoder`, `random_patch_masking` | Reconstruction, masked-pixel loss, mask, masked input |

Seven implementation files, plus `__init__.py`, this README, requirements,
`check.py`, and the two files in `assets/`. Submission runners import these
implementations directly; redundant former research-path shims are not shipped.

Pathfinder accepts `[B,1,128,128]` floating-point images in `[-1,1]`.
Connectivity output is **two logits** `[B,2]`, not a probability or sigmoid.
Plain/residual multi-task output order is `(logits, betti0, delta_betti0)`;
both scalar heads are instantiated, but the runner chooses the supervised head.
ResNet/Swin multi-task output is `(logits, auxiliary)`.
Scalar predictions use Softplus and have shape `[B]`, except the historical
capacity-Swin interface, which returns `[B,1]`. Keep this distinction when
loading existing state dictionaries or adapting a loss.

DRIVE uses the loader's normalised RGB `[B,3,512,512]` images. The reported
backbones are `plain`, `cnn_s`, `resnet18`, and `resnet152`. The standard forward
returns `logits_seg`, `logits_conn`, `pred_delta_b0`, `pred_delta_b1`,
`pred_global_betti`, and `feat_pool`. The DRIVE runner selects the losses;
it does not use the optional pixel-aligned or region-mask branches.

## Minimal use

From the directory containing `models/`:

```python
import torch
from models.plain import MultiTaskPlain
from models.swin import MultiTaskSwinT
from models.drive_unet import MultiTaskUNet

model = MultiTaskPlain(width=32).eval()
with torch.no_grad():
    logits, global_count, relative_count = model(torch.zeros(2, 1, 128, 128))

# Explicitly disable pretrained weights for offline construction.
swin = MultiTaskSwinT(pretrained=False)
vessel_model = MultiTaskUNet(backbone="resnet18", pretrained=False)
```

Use `pretrained=True` only for the corresponding ImageNet-initialised experiment.
It uses the torchvision cache and otherwise downloads official weights:
ResNet-18 `DEFAULT`, ResNet-50/152 `IMAGENET1K_V2`, Swin-T `DEFAULT`.
The one-channel stem averages the pretrained RGB weights. Scratch and pretrained
Swin preserve their original stochastic-depth settings (0 and 0.2 respectively).
Optional legacy `timm` paths are not required or checked for the paper workflow.
These dependency ranges are not the historical training-environment lockfile.

`HGRU()` uses the byte-exact upstream Gabor bank, 25 channels and eight recurrent
steps. The third-party Gabor file is not redistributed; obtain it separately as
described in [`assets/HGRU_ASSET.md`](assets/HGRU_ASSET.md). Missing or modified
files raise an error and never silently switch to random filters. SHA-256 is
checked before loading the upstream pickled NumPy dictionary.
The original random hidden-state initialisation occurs even in evaluation mode;
control the PyTorch RNG for reproducible comparisons. Do not substitute CNN
training settings for hGRU's prescribed protocol. `MultiTaskHGRU` remains for
compatibility, but the paper baseline uses `HGRU` without an auxiliary target.

## Checks and limits

```bash
python -I -B models/check.py --backward
python -I -B models/check.py --full-resolution
```

Both commands are offline CPU checks: no dataset, weight download or optimiser
step. The first checks 20 model configurations with small inputs and gradients;
the second uses Pathfinder 128x128 and DRIVE 512x512 inputs. Both verify output
contracts, finite tensors and a strict in-memory `state_dict` round trip.

The default relative Plain/residual models contain 656,804 / 495,540 parameters;
the small `width=6` Plain / `width=3` residual contain 23,470 / 17,881. Label-only
hGRU contains 285,637. Full counts are emitted by the check command. Capacity-Swin
configuration names retain historical approximate size labels; use measured
counts, not those labels, for reporting.

Relocation checks compare against the **current source snapshot**, not recovered
historical trained checkpoints. Existing state-dictionary names, constructor RNG
order and outputs are preserved; full-module pickle compatibility is not promised.
In particular, current DRIVE U-Nets retain an unused 65-parameter RoI head and
other historical shared branches. Their total counts differ from some older
run records. Do not remove heads, use `strict=False` to conceal mismatches, or
claim historical checkpoint equivalence until the training-source provenance is
resolved. See [reproduction notes](../docs/REPRODUCTION_NOTES.md).
