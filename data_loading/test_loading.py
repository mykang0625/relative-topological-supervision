"""CPU-only fixture tests. No downloads, generator imports or model training."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loading import pathfinder as pf
from data_loading import drive as dr


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pathfinder_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps({"qati_target_version": pf.V2}), encoding="utf-8")
    rows = []
    for index in range(4):
        path = f"train/dashed_with_points/scene_{index}.png"
        image_path = root / path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((128, 128), dtype=np.uint8)
        image[20:26, 30 + index:37 + index] = 255
        Image.fromarray(image).save(image_path)
        rows.append(dict(sample_id=str(index), split="train", image_path=path,
                         dashed_with_points_path=path, label=str(index % 2),
                         delta_betti0_ssl=str(index), betti0_initial_ssl="40",
                         betti0_bridged_ssl=str(40 - index), betti0="5",
                         p1_ssl_yx="[22, 33]", p2_ssl_yx="[70, 90]"))
    write_rows(root / "metadata_dashed_with_points/train_4.csv", rows)
    return rows


def drive_fixture(root):
    metadata = root / "metadata_qati_pairs_v1"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "summary.json").write_text(json.dumps({
        "target_version": "drive_native_analytic_capsule_pixel_centres_r6_v1"}), encoding="utf-8")
    for split in ("train", "val", "test"):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[..., 1] = 80
        vessel = np.zeros((48, 64), dtype=np.uint8)
        vessel[12:37, 20:42] = 255
        fov = np.full((48, 64), 255, dtype=np.uint8)
        for name, array in (("image", rgb), ("vessel", vessel), ("fov", fov)):
            Image.fromarray(array).save(root / f"{split}_{name}.png")
        rows = [dict(pair_id=f"{split}_{i}", image_id=split, split=split,
                     image_path=f"{split}_image.png", vessel_mask_path=f"{split}_vessel.png",
                     fov_mask_path=f"{split}_fov.png", label=str(i % 2),
                     p1_y="16", p1_x="24", p2_y="30", p2_x="36",
                     betti0="2", betti1="3", delta_betti0=str(i), delta_betti1="-1")
                for i in range(2)]
        write_rows(metadata / f"{split}_pairs.csv", rows)


class PathfinderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = pathfinder_fixture(self.root)
        self.csv = self.root / "metadata_dashed_with_points/train_4.csv"

    def test_shapes_values_and_distinct_betti_conventions(self):
        x, y, d, b, c = pf.load_split_data(self.root, "train_4")
        self.assertEqual(tuple(x.shape), (4, 128, 128))
        self.assertEqual(x.dtype, torch.uint8)
        self.assertEqual(y.dtype, torch.int64)
        self.assertEqual(d.dtype, torch.float32)
        self.assertEqual(y.tolist(), [0, 1, 0, 1])
        self.assertEqual(d.tolist(), [0, 1, 2, 3])
        self.assertEqual(b.tolist(), [40] * 4)
        self.assertEqual(c.shape, (4, 2, 2))
        self.assertEqual(pf.load_scaling_split(self.root, "train_4")[3].tolist(), [5] * 4)
        norm = pf.normalise_images(x)
        self.assertEqual(tuple(norm.shape), (4, 1, 128, 128))
        self.assertEqual((norm.min().item(), norm.max().item()), (-1, 1))

    def test_label_only_and_spt_do_not_require_targets(self):
        keep = {"sample_id", "split", "image_path", "label", "dashed_with_points_path"}
        write_rows(self.csv, [{k: v for k, v in row.items() if k in keep} for row in self.rows])
        self.assertEqual(len(pf.load_label_split(self.root, "train_4")), 2)
        self.assertTrue(torch.equal(pf.load_raw_images(self.root, "train_4"),
                                    pf.load_label_split(self.root, "train_4")[0]))
        with self.assertRaises(ValueError):
            pf.load_split_data(self.root, "train_4")

    def test_missing_or_legacy_version(self):
        for version in (None, "qati_v1", "unknown"):
            (self.root / "summary.json").write_text(json.dumps({"target_version": version}), encoding="utf-8")
            with self.assertRaises(ValueError):
                pf.load_split_data(self.root, "train_4")
        (self.root / "summary.json").write_text(json.dumps({"target_version": pf.DENSE_LEGACY}), encoding="utf-8")
        self.assertEqual(pf.read_split(self.root, "train_4")[1], pf.DENSE_LEGACY)

    def test_missing_targets_never_fall_back(self):
        for value in ("", "nan", "inf", "-1", "1.5"):
            self.rows[0]["delta_betti0_ssl"] = value
            write_rows(self.csv, self.rows)
            with self.assertRaises(ValueError):
                pf.load_split_data(self.root, "train_4")

    def test_inconsistent_target(self):
        self.rows[0]["betti0_bridged_ssl"] = "20"
        write_rows(self.csv, self.rows)
        with self.assertRaises(ValueError):
            pf.load_split_data(self.root, "train_4")

    def test_wrong_label(self):
        self.rows[0]["label"] = "2"
        write_rows(self.csv, self.rows)
        with self.assertRaises(ValueError):
            pf.load_split_data(self.root, "train_4")

    def test_bad_coordinates(self):
        for value in ("[1]", "[1,128]", "[NaN,2]"):
            self.rows[0]["p1_ssl_yx"] = value
            write_rows(self.csv, self.rows)
            with self.assertRaises(ValueError):
                pf.load_split_data(self.root, "train_4")

    def test_missing_image(self):
        self.rows[0]["image_path"] = self.rows[0]["dashed_with_points_path"] = "missing.png"
        write_rows(self.csv, self.rows)
        with self.assertRaises(FileNotFoundError):
            pf.load_split_data(self.root, "train_4")

    def test_wrong_image_shape(self):
        Image.new("L", (64, 64)).save(self.root / self.rows[0]["image_path"])
        with self.assertRaises(ValueError):
            pf.load_split_data(self.root, "train_4")

    def test_wrong_view_or_view_path(self):
        with self.assertRaises(ValueError):
            pf.load_split_data(self.root, "train_4", view="solid_with_points")
        self.rows[0]["dashed_with_points_path"] = "wrong.png"
        write_rows(self.csv, self.rows)
        with self.assertRaises(ValueError):
            pf.load_split_data(self.root, "train_4")

    def test_split_membership_size_and_duplicates(self):
        for rows in (self.rows[:3], self.rows[:3] + [self.rows[0]],
                     [{**self.rows[0], "split": "test"}, *self.rows[1:]]):
            write_rows(self.csv, rows)
            with self.assertRaises(ValueError):
                pf.load_split_data(self.root, "train_4")

    def test_split_not_found_no_size_fallback(self):
        for split in ("train_16", "train_32"):
            with self.assertRaises(FileNotFoundError):
                pf.load_split_data(self.root, split)
        with self.assertRaises(ValueError):
            pf.load_split_data(self.root, "../train_4")

    def test_path_escape_and_absolute(self):
        for relative in ("../escape.png", "C:/escape.png", "/escape.png"):
            with self.assertRaises(ValueError):
                pf.resolve_file(self.root, relative)

    def test_read_only(self):
        before = hashlib.sha256(self.csv.read_bytes()).digest()
        pf.load_split_data(self.root, "train_4")
        self.assertEqual(before, hashlib.sha256(self.csv.read_bytes()).digest())

    def test_shared_batch_augmentation_and_rng(self):
        x = pf.normalise_images(pf.load_raw_images(self.root, "train_4")[:1]).repeat(2, 1, 1, 1)
        a = torch.Generator().manual_seed(23)
        b = torch.Generator().manual_seed(23)
        for _ in range(16):
            y = pf.apply_augmentation(x, "dihedral_shift", a)
            self.assertTrue(torch.equal(y[0], y[1]))
            self.assertTrue(torch.equal(y, pf.apply_augmentation(x, "dihedral_shift", b)))
            self.assertEqual((y > 0).sum().item(), (x > 0).sum().item())
        with self.assertRaises(ValueError):
            pf.apply_augmentation(x, "typo", a)


class DriveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        drive_fixture(self.root)
        self.spec = dr.DRIVEPairDataSpec(dataset_root=self.root, image_size=64,
                                       num_workers=0, pin_memory=False, batch_size=2,
                                       max_shift=4)

    def test_shapes_types_and_signed_target(self):
        loaders = dr.build_drive_pair_dataloaders(self.spec)
        batch = next(iter(loaders["val"]))
        self.assertEqual(tuple(batch["image"].shape), (2, 3, 64, 64))
        self.assertEqual(tuple(batch["vessel_mask"].shape), (2, 1, 64, 64))
        self.assertEqual(batch["fov_mask"].dtype, torch.bool)
        self.assertEqual(tuple(batch["coordinates_yx"].shape), (2, 2, 2))
        self.assertEqual(batch["delta_betti1"].tolist(), [-1, -1])

    def test_eval_is_deterministic_and_train_is_seedable(self):
        for split in ("val", "test"):
            ds = dr.DRIVEPairDataset(self.spec, split)
            self.assertEqual(ds.augmentation, "none")
            self.assertTrue(torch.equal(ds[0]["image"], ds[0]["image"]))
        train = dr.DRIVEPairDataset(self.spec, "train")
        random.seed(23)
        a = train[0]
        random.seed(23)
        b = train[0]
        for key in ("image", "vessel_mask", "coordinates_yx"):
            self.assertTrue(torch.equal(a[key], b[key]))

    def test_joint_transform_alignment(self):
        vessel = np.zeros((32, 32), dtype=bool)
        vessel[12, 15] = True
        rgb = np.repeat(vessel[..., None], 3, axis=2).astype(np.uint8) * 255
        with patch.object(dr.random, "random", side_effect=[0, 0, 0]), \
             patch.object(dr.random, "randint", side_effect=[2, -3]):
            image, mask, fov, points = dr._joint_augmentation(
                rgb, vessel, vessel, [(12, 15)], augmentation="dihedral_shift", max_shift=4)
        y, x = map(int, points[0])
        self.assertTrue(mask[y, x])
        self.assertEqual(image[y, x, 0], 255)
        self.assertTrue(np.array_equal(mask, fov))

    def test_markers_are_baked_after_transform(self):
        ds = dr.DRIVEPairDataset(self.spec, "train")
        random.seed(7)
        item = ds[0]
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        white = (1 - mean) / std
        for point in item["coordinates_yx"]:
            y, x = torch.round(point * 63).long()
            self.assertTrue(torch.allclose(item["image"][:, y, x], white))

    def test_no_wrap_translation(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        image[3, 3] = 255
        self.assertEqual(dr.translate_without_wrap(image, 1, 1).sum(), 0)

    def test_leakage_is_rejected(self):
        path = self.root / "metadata_qati_pairs_v1/test_pairs.csv"
        rows = pf.read_rows(path, {"image_id"})
        for row in rows:
            row["image_id"] = "train"
        write_rows(path, rows)
        with self.assertRaises(ValueError):
            dr.build_drive_pair_dataloaders(self.spec)

    def test_missing_global_column(self):
        path = self.root / "metadata_qati_pairs_v1/train_pairs.csv"
        rows = pf.read_rows(path, {"betti0"})
        for row in rows:
            del row["betti0"]
        write_rows(path, rows)
        with self.assertRaises(ValueError):
            dr.DRIVEPairDataset(self.spec, "train")

    def test_shape_mismatch(self):
        Image.new("L", (32, 32)).save(self.root / "train_fov.png")
        with self.assertRaises(ValueError):
            dr.DRIVEPairDataset(self.spec, "train")

    def test_metadata_not_rewritten_by_shuffled_control(self):
        path = self.root / "metadata_qati_pairs_v1/train_pairs.csv"
        before = path.read_bytes()
        loaders = dr.build_drive_pair_dataloaders(self.spec)
        loaders["train"].dataset.rows[0]["delta_betti0"] = "99"
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(loaders["val"].dataset[0]["delta_betti0"].item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
