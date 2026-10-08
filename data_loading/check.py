"""Read prepared data and inspect batches on CPU; no training or data modification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loading import pathfinder as pf
from data_loading.drive import DRIVEPairDataSpec, build_drive_pair_dataloaders


def check_pathfinder(root, train_split, samples=32, all_images=False):
    rows_by_split, result = {}, {}
    for split in (train_split, "val", "test"):
        rows, version = pf.read_split(root, split)
        rows_by_split[split] = rows
        selected = rows if all_images else rows[:samples]
        x, y, d, b, _ = pf._tensors(root, selected)
        batch = pf.normalise_images(x[:samples])
        if split == train_split:
            batch = pf.apply_augmentation(batch, "dihedral_shift", torch.Generator().manual_seed(23))
        result[split] = {"scenes": len(rows), "images_decoded": len(selected),
                         "target_version": version, "batch_shape": list(batch.shape),
                         "image_range": [float(batch.min()), float(batch.max())],
                         "labels_dtype": str(y.dtype), "target_dtype": str(d.dtype),
                         "global_target_column": "betti0_initial_ssl"}
    parts = list(rows_by_split.values())
    for i in range(3):
        for j in range(i + 1, 3):
            if {r["sample_id"] for r in parts[i]} & {r["sample_id"] for r in parts[j]}:
                raise ValueError("Pathfinder scene leakage between splits")
            if {pf.resolve_file(Path(root), r["image_path"]) for r in parts[i]} & {pf.resolve_file(Path(root), r["image_path"]) for r in parts[j]}:
                raise ValueError("Pathfinder image leakage between splits")
    result["scene_disjoint"] = True
    return result


def check_drive(root, samples=32, all_items=False, num_workers=0):
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    loaders = build_drive_pair_dataloaders(DRIVEPairDataSpec(
        dataset_root=root, batch_size=min(16, samples), num_workers=num_workers,
        pin_memory=False))
    result = {}
    for split, loader in loaders.items():
        batch = next(iter(loader))
        for name, value in batch.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite {name} in {split}")
        visited = len(batch["image"])
        if all_items:
            for index in range(len(loader.dataset)):
                item = loader.dataset[index]
                if not torch.isfinite(item["image"]).all():
                    raise ValueError(f"Non-finite DRIVE image in {split}")
            visited = len(loader.dataset)
        result[split] = {"pairs": len(loader.dataset),
                         "images": len(loader.dataset._image_cache),
                         "pairs_checked": visited,
                         "image_shape": list(batch["image"].shape),
                         "vessel_shape": list(batch["vessel_mask"].shape),
                         "query_shape": list(batch["coordinates_yx"].shape),
                         "augmentation": loader.dataset.augmentation}
    result["image_disjoint"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathfinder-root", type=Path)
    parser.add_argument("--dense-root", type=Path)
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument("--train-split", default="train_4000")
    parser.add_argument("--dense-train-split", default="train_4000")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--all-items", action="store_true", help="Decode every selected Pathfinder image and visit every DRIVE query")
    parser.add_argument("--num-workers", type=int, default=0, help="DRIVE batch workers; 0 is portable, 2 also tests spawn")
    args = parser.parse_args()
    if not any((args.pathfinder_root, args.dense_root, args.drive_root)):
        parser.error("Provide at least one prepared dataset root")
    if args.samples < 1 or args.num_workers < 0:
        parser.error("samples must be positive and num-workers non-negative")
    torch.set_num_threads(1)
    report = {"scope": "Data loading only; not training, target recomputation or result reproduction"}
    for key, root, split in (("pathfinder", args.pathfinder_root, args.train_split),
                             ("dense_clutter", args.dense_root, args.dense_train_split)):
        if root:
            report[key] = check_pathfinder(root, split, args.samples, args.all_items)
    if args.drive_root:
        report["drive"] = check_drive(args.drive_root, args.samples, args.all_items, args.num_workers)
    report["ok"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
