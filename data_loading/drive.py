"""DRIVE Retinal Vessel Dataset loader with dynamic query-anchored marker baking.

The immutable RGB, field-of-view, and vessel rasters are shared across query pairs per image.
A query pair is rendered dynamically on-the-fly when `__getitem__` is called, after
the RGB image, masks, and query coordinates receive consistent dihedral and shift augmentations.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .pathfinder import resolve_file, read_rows, number

DRIVE_DATASET_ROOT = Path("datasets/DRIVE")
DRIVE_METADATA_NAME = "metadata_qati_pairs_v1"


def parse_augmentation(name: str) -> set[str]:
    presets = {
        "none": set(), "flips": {"flips"},
        "dihedral": {"flips", "transpose"}, "shift": {"flips", "shift"},
        "dihedral_shift": {"flips", "transpose", "shift"},
    }
    if name not in presets:
        raise ValueError(f"Unknown augmentation: {name}")
    return presets[name]


def translate_without_wrap(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate an array with zero fill and no wraparound."""
    array = np.asarray(mask)
    height, width = array.shape[-2:]
    output = np.zeros_like(array)

    src_y0 = max(0, -int(dy))
    src_y1 = min(height, height - int(dy))
    src_x0 = max(0, -int(dx))
    src_x1 = min(width, width - int(dx))
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return output

    dst_y0 = src_y0 + int(dy)
    dst_y1 = src_y1 + int(dy)
    dst_x0 = src_x0 + int(dx)
    dst_x1 = src_x1 + int(dx)
    output[..., dst_y0:dst_y1, dst_x0:dst_x1] = array[
        ..., src_y0:src_y1, src_x0:src_x1
    ]
    return output


def _resize_rgb(array: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(array)
    return np.asarray(
        image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8
    ).copy()


def _resize_binary(array: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8) * 255)
    return (
        np.asarray(image.resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8)
        > 0
    )


def _hflip(
    arrays: Sequence[np.ndarray], points: list[tuple[float, float]]
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    width = arrays[0].shape[1]
    return (
        [np.fliplr(array).copy() for array in arrays],
        [(y, width - 1 - x) for y, x in points],
    )


def _vflip(
    arrays: Sequence[np.ndarray], points: list[tuple[float, float]]
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    height = arrays[0].shape[0]
    return (
        [np.flipud(array).copy() for array in arrays],
        [(height - 1 - y, x) for y, x in points],
    )


def _transpose(
    arrays: Sequence[np.ndarray], points: list[tuple[float, float]]
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    transformed = []
    for array in arrays:
        if array.ndim == 3:
            transformed.append(np.transpose(array, (1, 0, 2)).copy())
        else:
            transformed.append(array.T.copy())
    return transformed, [(x, y) for y, x in points]


def _joint_augmentation(
    rgb: np.ndarray,
    vessel: np.ndarray,
    fov: np.ndarray,
    points: list[tuple[float, float]],
    *,
    augmentation: str,
    max_shift: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]]]:
    operations = parse_augmentation(augmentation)
    arrays: list[np.ndarray] = [rgb, vessel, fov]
    transformed_points = list(points)
    if "flips" in operations:
        if random.random() < 0.5:
            arrays, transformed_points = _hflip(arrays, transformed_points)
        if random.random() < 0.5:
            arrays, transformed_points = _vflip(arrays, transformed_points)
    if "transpose" in operations and random.random() < 0.5:
        arrays, transformed_points = _transpose(arrays, transformed_points)
    if "shift" in operations and max_shift > 0:
        dy = random.randint(-max_shift, max_shift)
        dx = random.randint(-max_shift, max_shift)
        shifted_rgb = translate_without_wrap(
            np.transpose(arrays[0], (2, 0, 1)), dy, dx
        )
        arrays = [
            np.transpose(shifted_rgb, (1, 2, 0)),
            translate_without_wrap(arrays[1], dy, dx),
            translate_without_wrap(arrays[2], dy, dx),
        ]
        transformed_points = [(y + dy, x + dx) for y, x in transformed_points]
    return arrays[0], arrays[1], arrays[2], transformed_points


def _bake_discs(
    rgb: np.ndarray,
    points: Sequence[tuple[float, float]],
    radius: float,
    colour: tuple[int, int, int],
) -> np.ndarray:
    output = rgb.copy()
    height, width = output.shape[:2]
    for y, x in points:
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(height, int(np.ceil(y + radius)) + 1)
        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(width, int(np.ceil(x + radius)) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
        patch = output[y0:y1, x0:x1]
        patch[disc] = np.asarray(colour, dtype=np.uint8)
    return output


def _normalise_rgb(rgb: np.ndarray, normalization: str) -> torch.Tensor:
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    if normalization == "zero_one":
        return tensor
    if normalization == "minus_one_to_one":
        return tensor * 2.0 - 1.0
    if normalization == "imagenet":
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype)[:, None, None]
        return (tensor - mean) / std
    raise ValueError(f"Unknown normalization: {normalization}")



@dataclass(frozen=True)
class DRIVEPairDataSpec:
    dataset_root: str | Path = DRIVE_DATASET_ROOT
    metadata_name: str = DRIVE_METADATA_NAME
    image_size: int = 512
    augmentation: str = "dihedral_shift"
    max_shift: int = 10
    bake_markers: bool = True
    marker_radius: float = 3.0
    marker_rgb: Tuple[int, int, int] = (255, 255, 255)
    normalization: str = "imagenet"
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    drop_last_train: bool = False

    def __post_init__(self) -> None:
        if self.image_size <= 1:
            raise ValueError("image_size must exceed one")
        if self.max_shift < 0:
            raise ValueError("max_shift must be non-negative")
        if self.marker_radius <= 0:
            raise ValueError("marker_radius must be positive")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("Invalid batch_size or num_workers")
        parse_augmentation(self.augmentation)
        if self.normalization not in {"imagenet", "zero_one", "minus_one_to_one"}:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        if Path(self.metadata_name).name != self.metadata_name or self.metadata_name in {".", ".."}:
            raise ValueError("metadata_name must be one directory name")

    @property
    def root(self) -> Path:
        return Path(self.dataset_root).resolve()

    @property
    def metadata_root(self) -> Path:
        return self.root / self.metadata_name


_resolve_under_root = resolve_file


class DRIVEPairDataset(Dataset[dict[str, Any]]):
    """Dataset providing items for real DRIVE retinal vessel images with query marker pairs."""

    def __init__(self, spec: DRIVEPairDataSpec, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.spec = spec
        self.split = split
        self.root = spec.root
        self.augmentation = spec.augmentation if split == "train" else "none"

        summary_path = spec.metadata_root / "summary.json"
        with summary_path.open("r", encoding="utf-8") as file:
            self.summary = json.load(file)
        metadata_path = spec.metadata_root / f"{split}_pairs.csv"
        self.rows = read_rows(metadata_path, {
            "pair_id", "image_id", "image_path", "fov_mask_path", "vessel_mask_path",
            "label", "p1_y", "p1_x", "p2_y", "p2_x", "betti0", "betti1",
            "delta_betti0", "delta_betti1",
        })
        if self.summary.get("target_version") != "drive_native_analytic_capsule_pixel_centres_r6_v1":
            raise ValueError("Missing/unsupported DRIVE target version; prepare targets first")
        seen_pairs, image_paths = set(), {}
        for row in self.rows:
            if row["pair_id"] in seen_pairs:
                raise ValueError(f"Duplicate DRIVE pair: {row['pair_id']}")
            seen_pairs.add(row["pair_id"])
            if "split" in row and row["split"] != split:
                raise ValueError(f"Split mismatch for {row['pair_id']}")
            if number(row, "label") not in (0, 1):
                raise ValueError(f"Invalid binary label: {row['pair_id']}")
            for key in ("betti0", "betti1", "delta_betti0", "delta_betti1"):
                number(row, key, signed=(key == "delta_betti1"))
            paths = tuple(str(resolve_file(self.root, row[k])) for k in
                          ("image_path", "vessel_mask_path", "fov_mask_path"))
            if row["image_id"] in image_paths and image_paths[row["image_id"]] != paths:
                raise ValueError(f"Inconsistent paths for image {row['image_id']}")
            image_paths[row["image_id"]] = paths
        # Pre-cache resized base rasters into memory (only 16-20 images per split, ~30MB total)
        self._image_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
        for row in self.rows:
            img_id = row["image_id"]
            if img_id not in self._image_cache:
                image_path = _resolve_under_root(self.root, row["image_path"])
                vessel_path = _resolve_under_root(self.root, row["vessel_mask_path"])
                fov_path = _resolve_under_root(self.root, row["fov_mask_path"])
                with Image.open(image_path) as image:
                    raw_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                with Image.open(vessel_path) as image:
                    raw_vessel = np.asarray(image.convert("L"), dtype=np.uint8) > 0
                with Image.open(fov_path) as image:
                    raw_fov = np.asarray(image.convert("L"), dtype=np.uint8) > 0

                if raw_rgb.shape[:2] != raw_vessel.shape or raw_fov.shape != raw_vessel.shape:
                    raise ValueError(f"RGB/mask shape mismatch for {img_id}")
                native_height, native_width = raw_vessel.shape
                if min(native_height, native_width) <= 1:
                    raise ValueError(f"Invalid native image shape for {img_id}")
                size = self.spec.image_size
                rgb = _resize_rgb(raw_rgb, size)
                vessel = _resize_binary(raw_vessel, size)
                fov = _resize_binary(raw_fov, size)
                scale_y = (size - 1) / (native_height - 1)
                scale_x = (size - 1) / (native_width - 1)
                self._image_cache[img_id] = (rgb, vessel, fov, scale_y, scale_x)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rgb_base, vessel_base, fov_base, scale_y, scale_x = self._image_cache[row["image_id"]]
        rgb = rgb_base.copy()
        vessel = vessel_base.copy()
        fov = fov_base.copy()
        size = self.spec.image_size

        native_points = np.asarray([[float(row["p1_y"]), float(row["p1_x"])],
                                    [float(row["p2_y"]), float(row["p2_x"])]])
        native_limits = np.asarray([(size - 1) / scale_y, (size - 1) / scale_x])
        if not np.isfinite(native_points).all() or np.any(native_points < 0) or np.any(native_points > native_limits):
            raise ValueError(f"Invalid query coordinates for {row['pair_id']}")
        points = [
            (float(row["p1_y"]) * scale_y, float(row["p1_x"]) * scale_x),
            (float(row["p2_y"]) * scale_y, float(row["p2_x"]) * scale_x),
        ]
        rgb, vessel, fov, points = _joint_augmentation(
            rgb,
            vessel,
            fov,
            points,
            augmentation=self.augmentation,
            max_shift=self.spec.max_shift,
        )
        for point in points:
            if not (0 <= point[0] < size and 0 <= point[1] < size):
                raise RuntimeError(
                    f"Augmented query left the canvas for {row['pair_id']}: {point}"
                )
        marker_free = rgb.copy()
        if self.spec.bake_markers:
            rgb = _bake_discs(
                rgb,
                points,
                self.spec.marker_radius,
                self.spec.marker_rgb,
            )

        coordinates_yx = torch.tensor(points, dtype=torch.float32) / float(size - 1)
        item: dict[str, Any] = {
            "image": _normalise_rgb(rgb, self.spec.normalization),
            "vessel_mask": torch.from_numpy(vessel[None].copy()).float(),
            "fov_mask": torch.from_numpy(fov[None].copy()).bool(),
            "coordinates_yx": coordinates_yx,
            "coordinates_xyxy": coordinates_yx[:, [1, 0]].reshape(4),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "betti0": torch.tensor(float(row["betti0"]), dtype=torch.float32),
            "betti1": torch.tensor(float(row["betti1"]), dtype=torch.float32),
            "delta_betti0": torch.tensor(
                float(row["delta_betti0"]), dtype=torch.float32
            ),
            "delta_betti1": torch.tensor(
                float(row["delta_betti1"]), dtype=torch.float32
            ),
            "pair_id": row["pair_id"],
            "image_id": row["image_id"],
        }
        if not self.spec.bake_markers:
            item["marker_free_image"] = _normalise_rgb(
                marker_free, self.spec.normalization
            )
        return item


def build_drive_pair_dataloaders(
    spec: DRIVEPairDataSpec,
) -> dict[str, DataLoader[dict[str, Any]]]:
    """Construct training, validation, and test loaders from a spec."""
    train_dataset = DRIVEPairDataset(spec, "train")
    val_dataset = DRIVEPairDataset(spec, "val")
    test_dataset = DRIVEPairDataset(spec, "test")

    datasets = (train_dataset, val_dataset, test_dataset)
    id_sets = [{row["image_id"] for row in ds.rows} for ds in datasets]
    path_sets = [{resolve_file(ds.root, row["image_path"]) for row in ds.rows} for ds in datasets]
    for i in range(3):
        for j in range(i + 1, 3):
            if id_sets[i] & id_sets[j] or path_sets[i] & path_sets[j]:
                raise ValueError("DRIVE train/val/test must be image-disjoint")
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=spec.batch_size,
            shuffle=True,
            num_workers=spec.num_workers,
            pin_memory=spec.pin_memory,
            drop_last=spec.drop_last_train,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            pin_memory=spec.pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            pin_memory=spec.pin_memory,
        ),
    }
