"""Read prepared dashed Pathfinder data without recomputing auxiliary targets.

The paper runners cache uint8 images on the CPU and augment normalised batches,
not individual Dataset items. Keep that order and the caller's RNG unchanged.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

V2 = "analytic_capsule_pixel_centres_r1p5_v2"
DENSE_LEGACY = "legacy_antialiased_bridge_prequantisation_v1"
VIEW = "dashed_with_points"


def resolve_file(root: Path, relative: str) -> Path:
    """Resolve a portable metadata path, refusing absolute paths and escapes."""
    root = root.resolve()
    if not relative or Path(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError(f"Expected a relative dataset path: {relative!r}")
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {relative!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError(f"Duplicate metadata columns: {path}")
        missing = required - set(fields)
        if missing:
            raise ValueError(f"Missing metadata columns {sorted(missing)}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty metadata split: {path}")
    for index, row in enumerate(rows, 2):
        if None in row or any(row.get(key) in (None, "") for key in required):
            raise ValueError(f"Incomplete metadata row {index}: {path}")
    return rows


def number(row: dict[str, str], key: str, *, signed: bool = False) -> float:
    value = float(row[key])
    if not np.isfinite(value) or value != int(value) or (not signed and value < 0):
        raise ValueError(f"Invalid integer target {key}={row[key]!r}")
    return value


def read_split(root: str | Path, split: str, view: str = VIEW,
               *, require_targets: bool = True) -> tuple[list[dict[str, str]], str | None]:
    """Read and validate metadata only; never select a different split or view.

    Only the paper's marker-visible dashed view is supported. The paired solid
    metadata carries dashed SSL columns too, so using them as solid targets would
    silently change the experimental question.
    """
    root = Path(root).resolve()
    if view != VIEW:
        raise ValueError(f"This release loader supports only {VIEW!r}; got {view!r}")
    if re.fullmatch(r"(?:train(?:_[1-9][0-9]*)?|val|test)", split) is None:
        raise ValueError(f"Unknown split: {split!r}")
    directory = root / f"metadata_{view}"
    if not directory.is_dir():
        directory = root / "metadata"
    # A missing size-specific CSV is an error, not permission to load another size.
    required = {"sample_id", "split", "image_path", "label"}
    version = None
    if require_targets:
        with (root / "summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        version = summary.get("qati_target_version", summary.get("target_version"))
        if version not in {V2, DENSE_LEGACY}:
            raise ValueError(f"Missing/unsupported target version {version!r}; prepare the dataset first")
        required |= {"delta_betti0_ssl", "betti0_initial_ssl", "betti0_bridged_ssl",
                     "betti0", "p1_ssl_yx", "p2_ssl_yx"}
    rows = read_rows(directory / f"{split}.csv", required)
    if split.startswith("train_") and len(rows) != int(split.split("_")[1]):
        raise ValueError(f"Split size does not match {split}: {len(rows)} rows")
    expected_split = "train" if split.startswith("train") else split
    ids, paths = set(), set()
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(f"Split membership mismatch for {row['sample_id']}")
        if number(row, "label") not in (0, 1):
            raise ValueError(f"Invalid binary label for {row['sample_id']}")
        selected = row.get(f"{view}_path", row["image_path"])
        if selected.replace("\\", "/") != row["image_path"].replace("\\", "/"):
            raise ValueError(f"Requested view and image_path disagree for {row['sample_id']}")
        image_path = resolve_file(root, selected)
        if row["sample_id"] in ids or image_path in paths:
            raise ValueError(f"Duplicate scene/image in {split}: {row['sample_id']}")
        ids.add(row["sample_id"])
        paths.add(image_path)
        if require_targets:
            delta = number(row, "delta_betti0_ssl")
            before = number(row, "betti0_initial_ssl")
            after = number(row, "betti0_bridged_ssl")
            number(row, "betti0")
            if before - after != delta:
                raise ValueError(f"Inconsistent stored target for {row['sample_id']}")
            for key in ("p1_ssl_yx", "p2_ssl_yx"):
                point = np.asarray(json.loads(row[key]), dtype=float)
                if point.shape != (2,) or not np.isfinite(point).all() or np.any((point < 0) | (point > 127)):
                    raise ValueError(f"Invalid query coordinates for {row['sample_id']}")
    return rows, version


def _images(root: str | Path, rows: list[dict[str, str]]) -> torch.Tensor:
    images = np.empty((len(rows), 128, 128), dtype=np.uint8)
    for index, row in enumerate(rows):
        with Image.open(resolve_file(Path(root), row["image_path"])) as image:
            if image.size != (128, 128):
                raise ValueError(f"Expected 128x128 image: {row['image_path']}")
            images[index] = np.asarray(image.convert("L"), dtype=np.uint8)
    return torch.from_numpy(images)


def load_split_data(root: str | Path, split: str, bridge_thickness: float = 3.0,
                    view: str = VIEW):
    """Return (uint8 images, int64 labels, float32 delta, raster beta0, query_yx).

    The fifth entry is a NumPy float32 (N,2,2) array in native pixel coordinates.
    It is diagnostic metadata: the classification model receives only the image.
    bridge_thickness remains for the original Plain runner API; no bridge is drawn.
    """
    if bridge_thickness != 3.0:
        raise ValueError("Targets are precomputed; bridge_thickness must remain 3.0")
    rows, _ = read_split(root, split, view)
    return _tensors(root, rows)


def _tensors(root, rows, *, beta_column="betti0_initial_ssl"):
    images = _images(root, rows)
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.int64)
    delta = torch.tensor([float(r["delta_betti0_ssl"]) for r in rows], dtype=torch.float32)
    beta = torch.tensor([float(r[beta_column]) for r in rows], dtype=torch.float32)
    coords = np.asarray([[json.loads(r[k]) for k in ("p1_ssl_yx", "p2_ssl_yx")]
                         for r in rows], dtype=np.float32)
    return images, labels, delta, beta, coords


def load_scaling_split(root: str | Path, split: str, view: str = VIEW):
    """Four-tensor API of scaling/pretrained runners; preserve their scene beta0.

    This fourth output is NOT the dashed-raster global target used by the Plain
    control. These historical runners train the relative head and do not use B.
    """
    rows, _ = read_split(root, split, view)
    return _tensors(root, rows, beta_column="betti0")[:4]


def load_label_split(root: str | Path, split: str, view: str = VIEW):
    """hGRU label-only interface: auxiliary columns are not required or returned."""
    rows, _ = read_split(root, split, view, require_targets=False)
    return _images(root, rows), torch.tensor([int(r["label"]) for r in rows], dtype=torch.int64)


def load_raw_images(root: str | Path, split: str = "train_4000", view: str = VIEW):
    """Unlabelled SPT input; no target computation and no augmentation here."""
    rows, _ = read_split(root, split, view, require_targets=False)
    return _images(root, rows)


def normalise_images(images: torch.Tensor) -> torch.Tensor:
    """Convert cached (N,128,128) uint8 images to float32 (N,1,128,128)."""
    if images.dtype != torch.uint8 or images.ndim != 3 or images.shape[1:] != (128, 128):
        raise ValueError("Expected uint8 images of shape (N,128,128)")
    return images.float().unsqueeze(1) / 127.5 - 1.0


def apply_augmentation(x: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
    """Original batch-shared transform and random draw order; input is in [-1,1]."""
    if mode not in {"none", "flips", "dihedral", "shift", "dihedral_shift"}:
        raise ValueError(f"Unknown augmentation: {mode}")
    if mode == "none":
        return x
    if torch.rand(1, generator=gen).item() < 0.5:
        x = torch.flip(x, [3])
    if torch.rand(1, generator=gen).item() < 0.5:
        x = torch.flip(x, [2])
    if mode == "flips":
        return x
    if "dihedral" in mode:
        if torch.rand(1, generator=gen).item() < 0.5:
            x = x.transpose(2, 3)
    if "shift" in mode:
        dy = int(torch.randint(-8, 9, (1,), generator=gen).item())
        dx = int(torch.randint(-8, 9, (1,), generator=gen).item())
        if dy != 0 or dx != 0:
            x = F.pad(x, (8, 8, 8, 8), mode="constant", value=-1.0)
            x = x[:, :, 8 + dy : 8 + dy + 128, 8 + dx : 8 + dx + 128]
    return x
