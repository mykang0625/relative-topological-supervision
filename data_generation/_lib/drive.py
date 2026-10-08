"""Prepare DRIVE sources and recompute targets at the frozen query coordinates."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
import argparse
import hashlib
from pathlib import Path
from typing import Any
from PIL import Image
from data_generation._lib.topology import betti_numbers, bridge_intervention, label_foreground
from data_generation._lib import common as pc
from concurrent.futures import ProcessPoolExecutor
from data_generation._lib.topology import betti_numbers, bridge_intervention, label_foreground


def _compute_fov_mask(rgb: np.ndarray) -> np.ndarray:
    """Compute robust retinal field of view mask."""
    fov = (rgb > 10).any(axis=-1)
    fov = ndi.binary_fill_holes(fov)
    lbl, n = ndi.label(fov)
    if n > 0:
        sizes = [int(np.sum(lbl == i)) for i in range(1, n + 1)]
        fov = (lbl == (np.argmax(sizes) + 1))
    return fov.astype(np.uint8) * 255


TARGET_VERSION = "drive_native_analytic_capsule_pixel_centres_r6_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _source_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # 1. Training images (IDs 21..40)
    train_records = []
    for img_id in range(21, 41):
        id_str = f"{img_id:02d}"
        img_path = root / "training" / "images" / f"{id_str}_training.png"
        vessel_path = root / "training" / "1st_manual" / f"{id_str}_manual1.png"
        fov_path = root / "training" / "mask" / f"{id_str}_training_mask.png"
        for p in (img_path, vessel_path, fov_path):
            if not p.is_file():
                raise FileNotFoundError(f"Missing required file: {p}")

        mask = _read_binary(vessel_path)
        topology = betti_numbers(mask)
        foreground_y, foreground_x = np.nonzero(mask)
        canvas_margin = int(
            min(
                foreground_y.min(),
                foreground_x.min(),
                mask.shape[0] - 1 - foreground_y.max(),
                mask.shape[1] - 1 - foreground_x.max(),
            )
        )
        train_records.append({
            "image_id": f"{id_str}_training",
            "image_path": img_path.relative_to(root).as_posix(),
            "fov_mask_path": fov_path.relative_to(root).as_posix(),
            "vessel_mask_path": vessel_path.relative_to(root).as_posix(),
            "height": mask.shape[0],
            "width": mask.shape[1],
            "betti0": topology.beta0,
            "betti1": topology.beta1,
            "vessel_pixels": int(mask.sum()),
            "vessel_fraction": float(mask.mean()),
            "canvas_margin_px": canvas_margin,
            "image_sha256": _sha256(img_path),
            "fov_mask_sha256": _sha256(fov_path),
            "vessel_mask_sha256": _sha256(vessel_path),
        })

    # 2. Test images (IDs 01..20)
    test_records = []
    for img_id in range(1, 21):
        id_str = f"{img_id:02d}"
        img_path = root / "test" / "images" / f"{id_str}_test.png"
        vessel_path = root / "test" / "1st_manual" / f"{id_str}_manual1.png"
        fov_path = root / "test" / "mask" / f"{id_str}_test_mask.png"
        for p in (img_path, vessel_path, fov_path):
            if not p.is_file():
                raise FileNotFoundError(f"Missing required file: {p}")

        mask = _read_binary(vessel_path)
        topology = betti_numbers(mask)
        foreground_y, foreground_x = np.nonzero(mask)
        canvas_margin = int(
            min(
                foreground_y.min(),
                foreground_x.min(),
                mask.shape[0] - 1 - foreground_y.max(),
                mask.shape[1] - 1 - foreground_x.max(),
            )
        )
        test_records.append({
            "image_id": f"{id_str}_test",
            "image_path": img_path.relative_to(root).as_posix(),
            "fov_mask_path": fov_path.relative_to(root).as_posix(),
            "vessel_mask_path": vessel_path.relative_to(root).as_posix(),
            "height": mask.shape[0],
            "width": mask.shape[1],
            "betti0": topology.beta0,
            "betti1": topology.beta1,
            "vessel_pixels": int(mask.sum()),
            "vessel_fraction": float(mask.mean()),
            "canvas_margin_px": canvas_margin,
            "image_sha256": _sha256(img_path),
            "fov_mask_sha256": _sha256(fov_path),
            "vessel_mask_sha256": _sha256(vessel_path),
        })

    return train_records, test_records


IDENTITY = "generation/drive_source.identity.json"


COMPLETE = "generation/drive_source.complete.json"


FOV_METHOD = "rgb_any_channel_gt10_fill_holes_largest_4connected_component"


def source_file(source, relative):
    path = source / relative
    candidates = [path.with_suffix(ext) for ext in (".png", ".tif", ".tiff", ".gif") if path.with_suffix(ext).is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one PNG/TIF/TIFF/GIF source for {relative}; found {len(candidates)}")
    return candidates[0]


def image_paths(m):
    return {r["row"][key]: r["pixels"][key] for r in m["images"]
            for key in ("image_path", "vessel_mask_path", "fov_mask_path")}


def verify_images(root, m, *, required=True):
    missing = []
    for relative, expected in image_paths(m).items():
        path = root / relative
        if not path.is_file():
            if required:
                raise ValueError(f"Missing DRIVE image/mask: {relative}")
            missing.append(relative)
        elif pc.image_record(path) != expected:
            raise ValueError(f"DRIVE reference pixels differ: {relative}")
    return missing


def source_identity(m, digest):
    return pc.json_bytes({"dataset_id": m["dataset_id"], "reference_sha256": digest,
                         "fov_construction": FOV_METHOD, "segmentation_labels": "existing_first_manual_annotation"})


def source_completion(m, digest):
    return pc.json_bytes({"reference_sha256": digest, "images": 40, "verified_rasters": 120,
                         "fov_construction": FOV_METHOD})


def verify_prepared(root, m, digest):
    pc.verify_payload(root, {IDENTITY: source_identity(m, digest), COMPLETE: source_completion(m, digest)}, required=True)
    verify_images(root, m)


def prepare(source, output, *, resume=False, dry_run=False, verify_only=False):
    output = pc.checked_root(output)
    source = Path(source).resolve() if source is not None else None
    if source is not None and (source == output or source in output.parents or output in source.parents):
        raise ValueError("Source and output must be separate, non-nested directories")
    m, digest = pc.load_reference("drive_reference")
    identity = source_identity(m, digest)
    complete = source_completion(m, digest)
    allowed = set(image_paths(m)) | {IDENTITY, COMPLETE, "generation/run.lock", "generation/drive_targets.identity.json", "generation/drive_targets.complete.json"}
    if output.exists() and any(output.iterdir()):
        if not (output / IDENTITY).is_file() or (output / IDENTITY).read_bytes() != identity:
            raise ValueError("Use a NEW output directory or a matching managed DRIVE directory")
    pc.inspect_files(output, allowed, extra_prefixes=("metadata_qati_pairs_v1/",))
    is_complete = (output / COMPLETE).exists()
    pc.verify_payload(output, {IDENTITY: identity, COMPLETE: complete})
    missing = verify_images(output, m, required=is_complete)
    if verify_only and not is_complete:
        raise ValueError("DRIVE source preparation is incomplete")
    if is_complete:
        print("DRIVE source pixels verified; no files changed", flush=True)
        return
    if (output / IDENTITY).exists() and not resume and not dry_run:
        raise ValueError("Interrupted DRIVE preparation; use --resume")
    if source is None or not source.is_dir():
        raise ValueError("Provide --source-root containing locally obtained DRIVE images and first manual annotations")
    from PIL import Image
    import numpy as np
    # Validate ALL source pixels and derived FOV before publishing any data.
    payload = {}
    for record in m["images"]:
        row = record["row"]
        with Image.open(source_file(source, row["image_path"])) as im:
            rgb = im.convert("RGB")
        with Image.open(source_file(source, row["vessel_mask_path"])) as im:
            vessel = im.convert("L")
        fov = Image.fromarray(_compute_fov_mask(np.asarray(rgb)))
        for key, image in (("image_path", rgb), ("vessel_mask_path", vessel), ("fov_mask_path", fov)):
            actual = {"mode": image.mode, "size": list(image.size), "pixels_sha256": pc.sha256(image.tobytes())}
            if actual != record["pixels"][key]:
                raise ValueError(f"Source pixels/FOV differ from the recorded experiment: {row[key]}")
            if row[key] in missing:
                payload[row[key]] = pc.png_bytes(image)
    print(f"DRIVE: validated 40 source images/annotations and derived FOV; {len(payload)} raster files missing", flush=True)
    if dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    with pc.exclusive_run(output):
        pc.inspect_files(output, allowed, extra_prefixes=("metadata_qati_pairs_v1/",))
        pc.write_new(output / IDENTITY, identity)
        for name, data in payload.items():
            path = output / name
            if path.exists():
                if pc.image_record(path) != image_paths(m)[name]:
                    raise ValueError(f"Existing image changed during preparation: {name}")
            else:
                pc.write_new(path, data)
        verify_images(output, m)
        pc.write_new(output / COMPLETE, complete)
    print("DRIVE source preparation complete; next run build_drive_targets.py", flush=True)


def prepare_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Locally obtained DRIVE root with training/ and test/; required except for verification")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.verify_only:
        parser.error("Select at most one inspection mode")
    try:
        prepare(args.source_root, args.output_root, resume=args.resume, dry_run=args.dry_run, verify_only=args.verify_only)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


TARGET_IDENTITY = "generation/drive_targets.identity.json"


TARGET_COMPLETE = "generation/drive_targets.complete.json"


META = "metadata_qati_pairs_v1"


def recompute_image(task):
    root_name, image, references = task
    import numpy as np
    root = Path(root_name)
    mask = _read_binary(root / image["vessel_mask_path"])
    labels, _ = label_foreground(mask)
    before = betti_numbers(mask)
    if before.beta0 != int(image["betti0"]) or before.beta1 != int(image["betti1"]):
        raise ValueError(f"Global Betti numbers differ: {image['image_id']}")
    results = []
    for reference in references:
        row = dict(reference)
        p1 = (int(row["p1_y"]), int(row["p1_x"]))
        p2 = (int(row["p2_y"]), int(row["p2_x"]))
        if not all(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] for y, x in (p1, p2)):
            raise ValueError("Query outside annotation")
        c1, c2 = int(labels[p1]), int(labels[p2])
        if not c1 or not c2:
            raise ValueError("Query does not lie on a vessel")
        result = bridge_intervention(mask, p1, p2, radius=6.0, before=before)
        row.update(label=int(c1 == c2), component1_id=c1, component2_id=c2,
                   distance_px=float(np.linalg.norm(np.asarray(p1, dtype=float) - p2)),
                   bridge_radius=6.0, betti0=before.beta0, betti1=before.beta1,
                   betti0_before=before.beta0, betti1_before=before.beta1,
                   betti0_after=result.after.beta0, betti1_after=result.after.beta1,
                   delta_betti0=result.delta_beta0, delta_betti1=result.delta_beta1)
        pc.compare_row(row, reference)
        # Preserve original formatting only after independently recomputing values.
        results.append(reference)
    return results


def make_metadata(root, m):
    actual_train, actual_test = _source_records(root)
    actual = {r["image_id"]: r for r in actual_train + actual_test}
    images = []
    for record in m["images"]:
        expected = record["row"]
        row = actual[expected["image_id"]]
        row["split"] = expected["split"]
        pc.compare_row(row, expected, ignored=("image_sha256", "fov_mask_sha256", "vessel_mask_sha256"))
        images.append(row)
    files = {}
    for split in ("train", "val", "test"):
        files[f"{META}/{split}_images.csv"] = pc.csv_bytes([r for r in images if r["split"] == split], m["image_columns"])
        files[f"{META}/{split}_pairs.csv"] = pc.csv_bytes([r for r in m["pairs"] if r["split"] == split], m["pair_columns"])
    files[f"{META}/images.csv"] = pc.csv_bytes(images, m["image_columns"])
    files[f"{META}/all_pairs.csv"] = pc.csv_bytes(m["pairs"], m["pair_columns"])
    summary = dict(m["summary"], query_selection="frozen_paper_pairs", target_source="manual_vessel_annotation",
                   fov_source="rgb_derived_not_official_fov")
    files[f"{META}/summary.json"] = pc.json_bytes(summary)
    return files


def validate_reference(m):
    all_ids = []
    for split, count in (("train", 16), ("val", 4), ("test", 20)):
        ids = m["splits"][split]
        if len(ids) != count or len(set(ids)) != count or set(ids) & set(all_ids):
            raise ValueError("Invalid/overlapping DRIVE image splits")
        all_ids.extend(ids)
    if len(m["pairs"]) != 2000 or len({r["pair_id"] for r in m["pairs"]}) != 2000:
        raise ValueError("Invalid/duplicate query pairs")
    for split, ids in m["splits"].items():
        for image_id in ids:
            rows = [r for r in m["pairs"] if r["image_id"] == image_id]
            if len(rows) != 50 or any(r["split"] != split or float(r["bridge_radius"]) != 6 for r in rows):
                raise ValueError("Invalid query membership/protocol")
    if {r["image_id"] for r in m["pairs"]} != set(all_ids):
        raise ValueError("Unexpected query image")


def build(root, *, workers=2, resume=False, dry_run=False, verify_only=False):
    root = pc.checked_root(root)
    m, digest = pc.load_reference("drive_reference")
    validate_reference(m)
    verify_prepared(root, m, digest)
    identity = pc.json_bytes({"reference_sha256": digest, "target_version": TARGET_VERSION,
                              "target_source": "manual_vessel_annotation", "queries": "frozen_paper_pairs"})
    files = make_metadata(root, m)
    completion = pc.json_bytes({"reference_sha256": digest, "recomputed_pairs": 2000,
                               "metadata_sha256": {p: pc.sha256(b) for p, b in files.items()}})
    allowed = set(files) | set(image_paths(m)) | {IDENTITY, COMPLETE, TARGET_IDENTITY, TARGET_COMPLETE, "generation/run.lock"}
    pc.inspect_files(root, allowed)
    pc.verify_payload(root, {TARGET_IDENTITY: identity, TARGET_COMPLETE: completion})
    done = (root / TARGET_COMPLETE).is_file()
    pc.verify_payload(root, files, required=done)
    if verify_only and not done:
        raise ValueError("DRIVE target generation is incomplete")
    if done and not verify_only:
        print("DRIVE targets/pixels verified; no files changed", flush=True)
        return
    if (root / TARGET_IDENTITY).exists() and not done and not resume and not dry_run:
        raise ValueError("Interrupted target generation; use --resume")
    if dry_run:
        print("DRIVE: validated source pixels and frozen 16/4/20 split, 2,000 pairs planned; no files written", flush=True)
        return
    tasks = [(str(root), record["row"], [r for r in m["pairs"] if r["image_id"] == record["row"]["image_id"]]) for record in m["images"]]

    def recompute():
        if workers == 1:
            for i, task in enumerate(tasks, 1):
                recompute_image(task)
                print(f"DRIVE targets checked: {i}/40 images", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, _ in enumerate(pool.map(recompute_image, tasks), 1):
                    print(f"DRIVE targets checked: {i}/40 images", flush=True)

    if verify_only:
        recompute()
        print("All 2,000 DRIVE targets independently recomputed and verified; no writes", flush=True)
        return
    with pc.exclusive_run(root):
        pc.write_new(root / TARGET_IDENTITY, identity)
        recompute()
        verify_prepared(root, m, digest)
        for path, data in files.items():
            pc.write_new(root / path, data)
        pc.write_new(root / TARGET_COMPLETE, completion)
    print("DRIVE auxiliary metadata complete: 16/4/20 images, 800/200/1000 pairs", flush=True)


def targets_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.dry_run and args.verify_only):
        parser.error("Positive workers and at most one inspection mode are required")
    try:
        build(args.dataset_root, workers=args.workers, resume=args.resume, dry_run=args.dry_run, verify_only=args.verify_only)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
