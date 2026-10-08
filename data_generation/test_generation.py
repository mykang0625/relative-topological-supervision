"""Compact safety/regression tests; synthetic fixtures are not paper results."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from data_generation._lib import pathfinder as sp
from unittest.mock import patch
from data_generation._lib import common as pc
from data_generation._lib import drive
from data_generation._lib import drive as targets
from data_generation._lib import dense_clutter as dense
import ast


def fixture():
    from PIL import Image, ImageDraw
    image = Image.new("L", (128, 128))
    draw = ImageDraw.Draw(image)
    draw.line((20, 32, 105, 32), fill=255, width=2)
    draw.line((20, 96, 105, 96), fill=255, width=2)
    for x, y in ((20, 32), (105, 96)):
        draw.ellipse((x-3, y-3, x+3, y+3), fill=255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    import numpy as np
    target = sp.compute_qati_v2_ssl_delta(np.asarray(image))
    records = []
    for i in range(8):
        sid = f"sample_{i:06d}"
        split = "train" if i < 6 else ("val" if i == 6 else "test")
        row = dict(sample_id=sid, split=split, image_path=f"{split}/dashed_with_points/{sid}.png",
                   label=str(i % 2), betti0="5", betti_0="5", betti0_class="0", bridge_betti0="4",
                   origin_yx="[32,20]", terminal_yx="[96,105]",
                   origin_yx_normalised="[0.0,0.0]", terminal_yx_normalised="[1.0,1.0]",
                   path_length="14", distractor_paths="3", total_paths="5", scene_seed=str(i+1), attempts="1")
        row.update({v + "_path": f"{split}/{v}/{sid}.png" for v in sp.VIEWS})
        row.update({k: json.dumps(v, separators=(",", ":")) if isinstance(v, list) else str(v) for k, v in target.items()})
        records.append({"row": row, "pixels_sha256": {v: sp.sha256(image.tobytes()) for v in sp.VIEWS}})
    ids = [r["row"]["sample_id"] for r in records]
    splits = {f"train_{n}": ids[:1] for n in (250, 500, 1000, 2000)}
    splits.update(train_4000=ids[:2], train_8000=ids[:3], train_16000=ids[:4], train_32000=ids[:6], val=[ids[6]], test=[ids[7]])
    manifest = dict(schema_version=1, dataset_id=sp.DATASET_ID, settings=sp.SETTINGS.copy(),
                    qati_target_version=sp.QATI_V2_TARGET_VERSION, columns=list(records[0]["row"]),
                    records=records, splits=splits)
    return manifest, png


class StagedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pathfinder-state-test-", dir=os.environ.get("PATHFINDER_TEST_TMP"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "data"
        self.m, self.png = fixture()
        self.digest = sp.sha256(sp.json_bytes(self.m))
        self.rendered = []

    def render(self, args):
        root, record = args
        self.rendered.append(record["row"]["sample_id"])
        for v in sp.VIEWS:
            sp.write_new(Path(root) / record["row"][v + "_path"], self.png)
        if record["pixels_sha256"] is None:
            hashes = {v: sp.pixel_hash(Path(root) / record["row"][v + "_path"]) for v in sp.VIEWS}
            sp.write_new(Path(root) / sp.pixel_record_name(record), sp.json_bytes(sp.pixel_record(record, hashes)))

    def run_stage(self, stage=4000, **kwargs):
        sp.run_stage(self.root, self.m, self.digest, stage, render=kwargs.pop("render", self.render), **kwargs)

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): sp.sha256(p.read_bytes())
                for p in self.root.rglob("*") if p.is_file()}

    def test_fixture_is_not_accepted_as_paper_manifest(self):
        sp.validate_manifest(self.m, paper=False)
        with self.assertRaises(ValueError):
            sp.validate_manifest(self.m)

    def test_dry_run_writes_nothing(self):
        self.run_stage(dry_run=True)
        self.assertFalse(self.root.exists())

    def test_stages_are_additive_and_idempotent(self):
        self.run_stage()
        before = self.snapshot()
        self.assertEqual(len(self.rendered), 4)
        self.run_stage(16000)
        self.assertEqual(len(self.rendered), 6)
        self.assertTrue(all(self.snapshot()[p] == h for p, h in before.items()))
        middle = self.snapshot()
        self.run_stage(32000)
        self.assertEqual(len(self.rendered), 8)
        self.assertTrue(all(self.snapshot()[p] == h for p, h in middle.items()))
        final = self.snapshot()
        self.run_stage(32000)
        self.run_stage(32000, verify_only=True)
        self.assertEqual(self.snapshot(), final)

    def test_expand_requires_previous_completed_stage(self):
        with self.assertRaisesRegex(ValueError, "First complete"):
            self.run_stage(16000)
        self.assertFalse(self.root.exists())
        self.run_stage()
        with self.assertRaisesRegex(ValueError, "First complete"):
            self.run_stage(32000)

    def test_unmanaged_directory_rejected_without_writes(self):
        self.root.mkdir()
        (self.root / "user.txt").write_text("untouched")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Unmanaged"):
            self.run_stage()
        self.assertEqual(before, self.snapshot())

    def test_resume_only_missing_scenes(self):
        def interrupted(args):
            self.render(args)
            raise RuntimeError("simulated interruption")
        with self.assertRaises(RuntimeError):
            self.run_stage(render=interrupted)
        self.assertFalse((self.root / "generation/stage_4000.complete.json").exists())
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "--resume"):
            self.run_stage()
        self.run_stage(resume=True)
        self.assertEqual(len(self.rendered), 4)
        self.assertTrue(all(self.snapshot()[p] == h for p, h in before.items()))

    def test_resume_partial_view(self):
        def interrupted(args):
            root, r = args
            sp.write_new(Path(root) / r["row"]["dashed_with_points_path"], self.png)
            raise RuntimeError("simulated interruption between views")
        with self.assertRaises(RuntimeError):
            self.run_stage(render=interrupted)
        before = self.snapshot()
        self.run_stage(resume=True)
        self.assertTrue(all(self.snapshot()[p] == h for p, h in before.items()))

    def test_corrupt_pixels_are_not_overwritten(self):
        self.run_stage()
        p = self.root / self.m["records"][0]["row"]["image_path"]
        from PIL import Image
        Image.new("L", (128, 128), 255).save(p)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Pixel checksum"):
            self.run_stage(16000, resume=True)
        self.assertEqual(before, self.snapshot())

    def test_missing_completed_image_is_not_silently_repaired(self):
        self.run_stage()
        (self.root / self.m["records"][0]["row"]["image_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing image"):
            self.run_stage(16000, resume=True)

    def test_metadata_change_rejected(self):
        self.run_stage()
        with (self.root / "metadata/train_4000.csv").open("ab") as f:
            f.write(b"unexpected row\n")
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.run_stage(16000)

    def test_wrong_protocol_rejected(self):
        self.run_stage()
        ident = self.root / "generation/dataset.json"
        value = json.loads(ident.read_text())
        value["qati_target_version"] = "old-v1"
        ident.write_bytes(sp.json_bytes(value))
        with self.assertRaisesRegex(ValueError, "different manifest/protocol"):
            self.run_stage(16000)

    def test_unknown_extra_image_rejected(self):
        self.run_stage()
        (self.root / "unexpected.png").write_bytes(self.png)
        with self.assertRaisesRegex(ValueError, "Unexpected file"):
            self.run_stage(16000)

    def test_missing_completed_summary_rejected(self):
        self.run_stage()
        (self.root / "summary.json").unlink()
        with self.assertRaisesRegex(ValueError, "summary/protocol"):
            self.run_stage(16000)

    def test_lock_only_crash_can_restart(self):
        with sp.exclusive_run(self.root):
            pass
        self.run_stage()
        self.assertTrue((self.root / "generation/stage_4000.complete.json").is_file())

    def test_verify_only_does_not_complete_partial_data(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_stage(verify_only=True)
        self.assertFalse(self.root.exists())

    def test_overlap_and_duplicate_manifest_ids_rejected(self):
        for mutate in (lambda m: m["records"].append(copy.deepcopy(m["records"][0])),
                       lambda m: m["splits"]["val"].append(m["splits"]["train_4000"][0])):
            modified = copy.deepcopy(self.m)
            mutate(modified)
            with self.assertRaises(ValueError):
                sp.validate_manifest(modified, paper=False)

    def test_write_new_preserves_different_existing_file(self):
        self.root.mkdir()
        path = self.root / "sample.txt"
        sp.write_new(path, b"original")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            sp.write_new(path, b"different")
        self.assertEqual(path.read_bytes(), b"original")

    def test_active_lock_rejects_second_writer(self):
        self.run_stage()
        before = self.snapshot()
        with sp.exclusive_run(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another generation"):
                self.run_stage(16000)
        self.assertEqual(before, self.snapshot())

    def test_extension_without_original_pixels_records_new_checksums(self):
        self.m["records"][5]["pixels_sha256"] = None
        self.digest = sp.sha256(sp.json_bytes(self.m))
        self.run_stage()
        self.run_stage(16000)
        before = self.snapshot()
        self.run_stage(32000)
        self.run_stage(32000, verify_only=True)
        self.assertTrue(all(self.snapshot()[p] == h for p, h in before.items()))
        saved = json.loads((self.root / "generation/stage_32000.complete.json").read_text())
        self.assertEqual(saved["metadata_matched_generated_scenes"], 1)
        record = self.root / sp.pixel_record_name(self.m["records"][5])
        record.write_bytes(record.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "completion record"):
            self.run_stage(32000)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="additional-preparation-test-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def make_drive(self, official=False):
        from PIL import Image
        import numpy as np
        source = self.base / "source"
        row = {"image_path": "training/images/21_training.png",
               "vessel_mask_path": "training/1st_manual/21_manual1.png",
               "fov_mask_path": "training/mask/21_training_mask.png"}
        rgb = Image.new("RGB", (9, 9), (20, 40, 80))
        vessel = Image.new("L", (9, 9), 255)
        for key, image in (("image_path", rgb), ("vessel_mask_path", vessel)):
            path = source / row[key]
            if official:
                path = path.with_suffix(".tif" if key == "image_path" else ".gif")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        fov = Image.fromarray(drive._compute_fov_mask(np.asarray(rgb)))
        pixels = {key: {"mode": im.mode, "size": list(im.size), "pixels_sha256": pc.sha256(im.tobytes())}
                  for key, im in (("image_path", rgb), ("vessel_mask_path", vessel), ("fov_mask_path", fov))}
        return source, {"dataset_id": "test-only", "images": [{"row": row, "pixels": pixels}]}

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): pc.sha256(p.read_bytes()) for p in root.rglob("*") if p.is_file()}

    def test_csv_and_semantic_comparison(self):
        pc.compare_row({"x": 1, "q": "[1,2]", "p": "a\\b"}, {"x": "1.0", "q": "[1, 2]", "p": "a/b"}, path_keys=("p",))
        with self.assertRaisesRegex(ValueError, "Metadata differs"):
            pc.compare_row({"target": 2}, {"target": "3"})
        self.assertEqual(pc.csv_bytes([{"a": "b"}], ["a"]), b"a\nb\n")

    def test_no_clobber(self):
        path = self.base / "record"
        pc.write_new(path, b"first")
        with self.assertRaises(ValueError):
            pc.write_new(path, b"second")
        self.assertEqual(path.read_bytes(), b"first")

    def test_drive_dry_run_no_writes(self):
        source, m = self.make_drive()
        output = self.base / "out"
        with patch.object(pc, "load_reference", return_value=(m, "test")):
            drive.prepare(source, output, dry_run=True)
        self.assertFalse(output.exists())

    def test_drive_official_tif_gif_and_idempotence(self):
        source, m = self.make_drive(official=True)
        output = self.base / "out"
        before_source = self.snapshot(source)
        with patch.object(pc, "load_reference", return_value=(m, "test")):
            drive.prepare(source, output)
            before = self.snapshot(output)
            drive.prepare(None, output, verify_only=True)
            drive.prepare(None, output)
            self.assertEqual(self.snapshot(output), before)
        self.assertEqual(self.snapshot(source), before_source)

    def test_drive_in_place_rejected(self):
        source, _ = self.make_drive()
        before = self.snapshot(source)
        with self.assertRaisesRegex(ValueError, "separate"):
            drive.prepare(source, source)
        self.assertEqual(self.snapshot(source), before)

    def test_drive_unmanaged_output_rejected(self):
        source, m = self.make_drive()
        output = self.base / "out"
        output.mkdir()
        (output / "user-file").write_text("keep")
        with patch.object(pc, "load_reference", return_value=(m, "test")):
            with self.assertRaisesRegex(ValueError, "NEW"):
                drive.prepare(source, output)
        self.assertEqual((output / "user-file").read_text(), "keep")

    def test_drive_wrong_source_pixels_fail_before_output(self):
        source, m = self.make_drive()
        m["images"][0]["pixels"]["image_path"]["pixels_sha256"] = "0" * 64
        output = self.base / "out"
        with patch.object(pc, "load_reference", return_value=(m, "test")):
            with self.assertRaisesRegex(ValueError, "Source pixels"):
                drive.prepare(source, output)
        self.assertFalse(output.exists())

    def test_drive_completed_corruption_rejected(self):
        source, m = self.make_drive()
        output = self.base / "out"
        with patch.object(pc, "load_reference", return_value=(m, "test")):
            drive.prepare(source, output)
            (output / m["images"][0]["row"]["image_path"]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing DRIVE"):
                drive.prepare(source, output, resume=True)

    def test_drive_partial_resume(self):
        source, m = self.make_drive()
        output = self.base / "out"
        output.mkdir()
        pc.write_new(output / drive.IDENTITY, drive.source_identity(m, "test"))
        with patch.object(pc, "load_reference", return_value=(m, "test")):
            with self.assertRaisesRegex(ValueError, "--resume"):
                drive.prepare(source, output)
            drive.prepare(source, output, resume=True)
        self.assertTrue((output / drive.COMPLETE).is_file())

    def test_drive_ambiguous_source_rejected(self):
        source, _ = self.make_drive()
        path = source / "training/images/21_training.png"
        path.with_suffix(".tif").write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            drive.source_file(source, "training/images/21_training.png")

    def test_reference_split_and_query_validation(self):
        m, _ = pc.load_reference("drive_reference")
        targets.validate_reference(m)
        altered = copy.deepcopy(m)
        altered["splits"]["val"][0] = altered["splits"]["train"][0]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            targets.validate_reference(altered)
        altered = copy.deepcopy(m)
        altered["pairs"][0]["bridge_radius"] = "3"
        with self.assertRaisesRegex(ValueError, "protocol"):
            targets.validate_reference(altered)

    def test_dense_dry_run_and_no_in_place_generation(self):
        output = self.base / "dense"
        dense.prepare(output, dry_run=True)
        self.assertFalse(output.exists())
        output.mkdir()
        (output / "user-file").write_text("keep")
        with self.assertRaisesRegex(ValueError, "NEW"):
            dense.prepare(output)

    def test_dense_reference_replay_bounds(self):
        with self.assertRaisesRegex(ValueError, "bounds"):
            dense.generate_single_scene(1, "train", 0, 1, self.base, start_attempt=-1)

    def test_payload_missing_and_changed_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing completed"):
            pc.verify_payload(self.base, {"needed": b"x"}, required=True)
        (self.base / "needed").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "Changed existing"):
            pc.verify_payload(self.base, {"needed": b"x"})


DATA = Path(__file__).resolve().parent


class LayoutTests(unittest.TestCase):
    def test_no_research_or_training_imports(self):
        forbidden = {"iclr2027", "src", "torch", "torchvision", "timm"}
        for path in DATA.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    self.assertNotIn(module.split(".")[0], forbidden, str(path))

    def test_six_public_commands(self):
        expected = {"prepare_4k.py", "extend_to_16k.py", "extend_to_32k.py",
                    "prepare_dense_clutter.py", "prepare_drive.py", "build_drive_targets.py"}
        self.assertEqual({p.name for p in DATA.glob("*.py") if p.name not in {"__init__.py", "test_generation.py", "check.py"}}, expected)

    def test_reference_files_and_vendor_licence(self):
        for name in ("pathfinder_v2", "dense_clutter_reference", "drive_reference"):
            self.assertTrue((DATA / "manifests" / (name + ".json.gz")).is_file())
            self.assertTrue((DATA / "manifests" / (name + ".json.gz.sha256")).is_file())
        self.assertIn("MIT License", (DATA / "vendor/LICENSE").read_text())


if __name__ == "__main__":
    unittest.main()
