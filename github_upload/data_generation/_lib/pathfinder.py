"""Image-derived V2 targets and additive 4K to 16K to 32K reproduction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from PIL import Image
from scipy import ndimage
from data_generation._lib.qati_target import qati_delta_from_foreground
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import io
import json
import os
import re
import tempfile
from data_generation._lib.scenes import GenerationSettings, SceneJob, VIEWS, generate_scene
from data_generation._lib.common import exclusive_run, json_bytes, sha256, write_new
from data_generation._lib.qati_target import QATI_V2_TARGET_VERSION, qati_delta_from_foreground


_Y, _X = np.ogrid[-4:5, -4:5]


_CIRCLE_KERNEL = (_X**2 + _Y**2 <= 3.0**2).astype(np.float32)


_CIRCLE_KERNEL /= _CIRCLE_KERNEL.sum()


def detect_query_markers_ssl(
    gray_norm: np.ndarray, threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect two query marker coordinates via disc template correlation and distance transform."""
    binary = gray_norm > threshold
    dt = ndimage.distance_transform_edt(binary)
    conv = ndimage.correlate(gray_norm, _CIRCLE_KERNEL, mode="constant", cval=0.0)
    score = conv * dt

    local_max = (score == ndimage.maximum_filter(score, size=7)) & (score > 0.3)
    coords = np.argwhere(local_max)

    if len(coords) >= 2:
        vals = np.array([score[c[0], c[1]] for c in coords])
        order = np.argsort(vals)[::-1]
        coords = coords[order]
        p1 = coords[0].astype(float)
        p2 = None
        for cand in coords[1:]:
            if np.linalg.norm(cand - p1) >= 6.0:
                p2 = cand.astype(float)
                break
        if p2 is None:
            p2 = coords[1].astype(float)
        return p1, p2
    elif len(coords) == 1:
        p1 = coords[0].astype(float)
        score_copy = score.copy()
        y, x = np.ogrid[:128, :128]
        score_copy[(y - p1[0])**2 + (x - p1[1])**2 <= 36] = 0
        p2 = np.array(np.unravel_index(np.argmax(score_copy), score.shape), dtype=float)
        return p1, p2
    else:
        p1 = np.array(np.unravel_index(np.argmax(score), score.shape), dtype=float)
        score_copy = score.copy()
        y, x = np.ogrid[:128, :128]
        score_copy[(y - p1[0])**2 + (x - p1[1])**2 <= 36] = 0
        p2 = np.array(np.unravel_index(np.argmax(score_copy), score.shape), dtype=float)
        return p1, p2


def compute_qati_v2_ssl_delta(
    img_input: Path | np.ndarray, bridge_thickness: float = 3.0, threshold: float = 0.1
) -> Dict[str, Any]:
    """Compute query-anchored QATI V2 target directly from the pixel raster."""
    if isinstance(img_input, (str, Path)):
        with Image.open(img_input) as im:
            gray_uint8 = np.asarray(im.convert("L"), dtype=np.uint8)
        gray_norm = gray_uint8.astype(np.float32) / 255.0
    else:
        if img_input.dtype == np.uint8:
            gray_norm = img_input.astype(np.float32) / 255.0
        else:
            gray_norm = np.clip(img_input.astype(np.float32), 0.0, 1.0)

    foreground = gray_norm > threshold
    p1, p2 = detect_query_markers_ssl(gray_norm, threshold=threshold)

    b0_initial, b0_bridged, _ = qati_delta_from_foreground(
        foreground, p1, p2, thickness=bridge_thickness
    )
    delta_v2 = max(0, b0_initial - b0_bridged)

    size = gray_norm.shape[0]
    p1_norm = [float(p1[0]) / max(1, size - 1), float(p1[1]) / max(1, size - 1)]
    p2_norm = [float(p2[0]) / max(1, size - 1), float(p2[1]) / max(1, size - 1)]

    return {
        "delta_betti0_ssl": int(delta_v2),
        "delta_betti0": int(delta_v2),
        "betti0_initial_ssl": int(b0_initial),
        "betti0_bridged_ssl": int(b0_bridged),
        "p1_ssl_yx": [float(p1[0]), float(p1[1])],
        "p2_ssl_yx": [float(p2[0]), float(p2[1])],
        "p1_ssl_normalised": p1_norm,
        "p2_ssl_normalised": p2_norm,
    }


ROOT = Path(__file__).resolve().parents[2]


DEFAULT_MANIFEST = ROOT / "data_generation/manifests/pathfinder_v2.json.gz"


STAGES = (4000, 16000, 32000)


SIZES = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000)


SETTINGS = dict(path_length=14, window_size=128, line_thickness=1.5,
                marker_radius=3.0, bridge_thickness=3.0,
                component_threshold=0.1, max_scene_attempts=2000)


DATASET_ID = "paper-pathfinder128-nested-v2"


RENDER_BACKEND = "scipy_grey_dilation_constant_v1"


POINT_COLUMNS = ("p1_ssl_yx", "p2_ssl_yx", "p1_ssl_normalised", "p2_ssl_normalised",
                 "origin_yx", "terminal_yx", "origin_yx_normalised", "terminal_yx_normalised")


TEXT_COLUMNS = {"sample_id", "split", "image_path", *(v + "_path" for v in VIEWS)}


def pixel_hash(path):
    """Hash decoded L pixels, not PNG compression/ancillary metadata."""
    from PIL import Image
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (128, 128):
            raise ValueError(f"Expected 128x128 grayscale image: {path}")
        return sha256(image.tobytes())


def same_row(actual, expected):
    if set(actual) != set(expected):
        return False
    for key in expected:
        a, b = actual[key], expected[key]
        if key in TEXT_COLUMNS:
            if str(a) != str(b):
                return False
        elif key in POINT_COLUMNS:
            if (json.loads(a) if isinstance(a, str) else a) != (json.loads(b) if isinstance(b, str) else b):
                return False
        elif float(a) != float(b):
            return False
    return True


def load_manifest(path=DEFAULT_MANIFEST, *, paper=True):
    path = Path(path)
    compressed = path.read_bytes()
    expected = path.with_suffix(path.suffix + ".sha256").read_text().strip()
    if sha256(compressed) != expected:
        raise ValueError("Reference manifest checksum mismatch")
    manifest = json.loads(gzip.decompress(compressed))
    validate_manifest(manifest, paper=paper)
    return manifest, expected


def validate_manifest(m, *, paper=True):
    if m["schema_version"] != 1 or m["qati_target_version"] != QATI_V2_TARGET_VERSION:
        raise ValueError("Unsupported manifest schema/target version")
    if m["settings"] != SETTINGS or m["dataset_id"] != DATASET_ID:
        raise ValueError("Manifest does not describe the fixed paper protocol")
    if len(m["columns"]) != 29 or len(set(m["columns"])) != 29:
        raise ValueError("Expected the 29-column paper metadata schema")
    records = m["records"]
    by_id = {r["row"]["sample_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate sample IDs")
    for sid, record in by_id.items():
        row = record["row"]
        if not re.fullmatch(r"sample_\d{6}", sid) or row["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid sample ID/split")
        if set(row) != set(m["columns"]):
            raise ValueError(f"Invalid row schema: {sid}")
        for view in VIEWS:
            if row[view + "_path"] != f"{row['split']}/{view}/{sid}.png":
                raise ValueError(f"Noncanonical image path: {sid}")
        if row["image_path"] != row["dashed_with_points_path"]:
            raise ValueError(f"Unexpected default view: {sid}")
        hashes = record["pixels_sha256"]
        if hashes is None:
            if row["split"] != "train" or int(sid.split("_")[1]) < 17800:
                raise ValueError("Only unavailable extension rasters may be metadata-only")
        elif set(hashes) != set(VIEWS) or any(
            not re.fullmatch(r"[0-9a-f]{64}", h) for h in record["pixels_sha256"].values()
        ):
            raise ValueError(f"Invalid image checksums: {sid}")
        if int(row["path_length"]) != 14 or int(row["label"]) not in (0, 1):
            raise ValueError(f"Invalid scene specification: {sid}")
        b = int(row["betti0"])
        if not 5 <= b <= 13 or int(row["distractor_paths"]) != b - 2:
            raise ValueError(f"Invalid distractor count: {sid}")
        if float(row["delta_betti0"]) != float(row["delta_betti0_ssl"]) or float(row["delta_betti0"]) != int(row["betti0_initial_ssl"]) - int(row["betti0_bridged_ssl"]):
            raise ValueError(f"Inconsistent V2 training target: {sid}")
    if set(m["splits"]) != {"val", "test", *(f"train_{n}" for n in SIZES)}:
        raise ValueError("Unexpected split list")
    previous = set()
    for n in SIZES:
        ids = m["splits"][f"train_{n}"]
        if len(ids) != len(set(ids)) or (paper and len(ids) != n) or not previous <= set(ids):
            raise ValueError(f"Invalid nested membership: train_{n}")
        if any(by_id[sid]["row"]["split"] != "train" for sid in ids):
            raise ValueError("Held-out scene in training")
        previous = set(ids)
    for split in ("val", "test"):
        ids = m["splits"][split]
        if len(ids) != len(set(ids)) or (paper and len(ids) != 900) or previous & set(ids):
            raise ValueError(f"Invalid/overlapping {split} membership")
        if any(by_id[sid]["row"]["split"] != split for sid in ids):
            raise ValueError(f"Wrong split assignment: {split}")
        previous.update(ids)
    if previous != set(by_id):
        raise ValueError("Orphan records in manifest")


def identity(m, digest):
    return {"schema_version": 1, "dataset_id": m["dataset_id"],
            "reference_manifest_sha256": digest, "qati_target_version": m["qati_target_version"],
            "settings": m["settings"], "render_backend": RENDER_BACKEND}


def selected_records(m, stage):
    ids = set(m["splits"][f"train_{stage}"] + m["splits"]["val"] + m["splits"]["test"])
    return [r for r in m["records"] if r["row"]["sample_id"] in ids]


def metadata_files(m, stage):
    """Add size-named tables; train.csv intentionally remains the main 4K split."""
    by_id = {r["row"]["sample_id"]: r["row"] for r in m["records"]}
    splits = {k: ids for k, ids in m["splits"].items()
              if not k.startswith("train_") or int(k.split("_")[1]) <= stage}
    splits["train"] = m["splits"]["train_4000"]
    files = {}
    for directory, view in [("metadata", "dashed_with_points")] + [("metadata_" + v, v) for v in VIEWS]:
        for name, ids in splits.items():
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=m["columns"], lineterminator="\n")
            writer.writeheader()
            for sid in ids:
                row = dict(by_id[sid])
                row["image_path"] = row[view + "_path"]
                writer.writerow(row)
            files[f"{directory}/{name}.csv"] = buffer.getvalue().encode()
    return files


def pixel_record_name(record):
    return f"generation/pixels/{record['row']['sample_id']}.json"


def pixel_record(record, hashes):
    return {"schema_version": 1, "sample_id": record["row"]["sample_id"],
            "reference_row_sha256": sha256(json_bytes(record["row"])),
            "reference_pixels_compared": False, "pixels_sha256": hashes}


def expected_pixels(root, record, *, allow_missing=False):
    if record["pixels_sha256"] is not None:
        return record["pixels_sha256"]
    path = root / pixel_record_name(record)
    if not path.is_file():
        if allow_missing:
            return None
        raise ValueError(f"Missing generated pixel record: {path.name}")
    saved = json.loads(path.read_bytes())
    hashes = saved.get("pixels_sha256", {})
    if (set(hashes) != set(VIEWS) or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values())
            or saved != pixel_record(record, hashes)):
        raise ValueError(f"Invalid generated pixel record: {path.name}")
    return hashes


def receipt(m, digest, stage, files, root=None):
    generated = [r for r in selected_records(m, stage) if r["pixels_sha256"] is None]
    if generated and root is None:
        raise ValueError("Generated-pixel records are required for this stage")
    return {**identity(m, digest), "stage": stage,
            "split_sizes": {"train": len(m["splits"][f"train_{stage}"]),
                            "val": len(m["splits"]["val"]), "test": len(m["splits"]["test"])},
            "verified_scenes": len(selected_records(m, stage)),
            "reference_pixel_scenes": len(selected_records(m, stage)) - len(generated),
            "metadata_matched_generated_scenes": len(generated),
            "generated_pixel_records_sha256": {pixel_record_name(r): sha256((root / pixel_record_name(r)).read_bytes()) for r in generated},
            "metadata_sha256": {name: sha256(content) for name, content in files.items()}}


def verify_target(root, record):
    row = record["row"]
    calculated = compute_qati_v2_ssl_delta(root / row["image_path"])
    for key, value in calculated.items():
        expected = json.loads(row[key]) if key in POINT_COLUMNS else float(row[key])
        if value != expected:
            raise ValueError(f"Image-derived V2 target/query mismatch: {row['sample_id']} {key}")


def verify_images(root, records, *, allow_missing=False, targets=False):
    missing = []
    for index, record in enumerate(records, 1):
        hashes = expected_pixels(root, record, allow_missing=allow_missing)
        absent = hashes is None
        for view in VIEWS:
            path = root / record["row"][view + "_path"]
            if not path.is_file():
                if not allow_missing:
                    raise ValueError(f"Missing image: {path}")
                absent = True
            elif hashes is not None and pixel_hash(path) != hashes[view]:
                raise ValueError(f"Pixel checksum mismatch: {path}; existing images are never overwritten")
        if absent:
            missing.append(record)
        elif targets:
            verify_target(root, record)
        if index % 2000 == 0:
            print(f"Verified {index}/{len(records)} scenes", flush=True)
    return missing


def inspect_existing(root, m, digest, stage):
    """Read-only preflight, also used for dry-run and verify-only."""
    if root.is_symlink():
        raise ValueError("Dataset root may not be a symlink")
    if not root.exists() or not any(root.iterdir()):
        return [], False
    ident = root / "generation/dataset.json"
    # A lock-only directory can result if the first process crashed before setup.
    all_files = list(root.rglob("*"))
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in all_files):
        raise ValueError("Symlinks/junctions are not allowed inside a generated dataset")
    if not ident.exists() and {p.relative_to(root).as_posix() for p in all_files if p.is_file()} == {"generation/run.lock"}:
        return [], False
    if not ident.is_file() or ident.read_bytes() != json_bytes(identity(m, digest)):
        raise ValueError("Unmanaged dataset or different manifest/protocol; choose a NEW output directory")
    files = metadata_files(m, stage)
    allowed = set(files) | {r["row"][v + "_path"] for r in selected_records(m, stage) for v in VIEWS}
    allowed |= {pixel_record_name(r) for r in selected_records(m, stage) if r["pixels_sha256"] is None}
    allowed |= {"summary.json", "generation/dataset.json", "generation/run.lock"}
    completed = []
    for n in STAGES:
        rel = f"generation/stage_{n}.complete.json"
        allowed.add(rel)
        path = root / rel
        if path.exists():
            if n > stage:
                raise ValueError(f"Dataset already extends to {n}; use that stage's --verify-only")
            required = metadata_files(m, n)
            if path.read_bytes() != json_bytes(receipt(m, digest, n, required, root)):
                raise ValueError(f"Invalid completion record: {rel}")
            for name, content in required.items():
                if not (root / name).is_file() or (root / name).read_bytes() != content:
                    raise ValueError(f"Missing/changed completed metadata: {name}")
            completed.append(n)
    if completed != list(STAGES[:len(completed)]):
        raise ValueError("Completion records are not a consecutive stage prefix")
    if completed and not (root / "summary.json").is_file():
        raise ValueError("Missing completed dataset summary/protocol")
    for path in all_files:
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith("generation/work/") or path.name.startswith(".publish-"):
            # Interrupted scratch files are never treated as completed samples.
            continue
        if rel not in allowed:
            raise ValueError(f"Unexpected file in managed dataset: {rel}")
        if rel in files and path.read_bytes() != files[rel]:
            raise ValueError(f"Different existing metadata: {rel}")
    if (root / "summary.json").exists() and (root / "summary.json").read_bytes() != json_bytes(summary(m, digest)):
        raise ValueError("Changed summary/protocol")
    return completed, True


def summary(m, digest):
    return {**identity(m, digest), "default_training_split": "train_4000",
            "views": list(VIEWS), "note": "Available counts are recorded in generation/stage_*.complete.json; train.csv always aliases train_4000."}


def render_record(args):
    """Generate in private scratch space; check BEFORE publishing missing files."""
    root_string, record = args
    root = Path(root_string)
    expected = record["row"]
    work = root / "generation/work"
    work.mkdir(parents=True, exist_ok=True)
    # The legacy module silently chooses OpenCV when installed and SciPy when
    # unavailable. They differ on the even-sized dilation footprints used here.
    # Pin the existing SciPy fallback semantics that match the frozen corpus.
    from types import SimpleNamespace
    import numpy as np
    from scipy import ndimage
    from data_generation._lib import rendering as pg

    def dilate(image, kernel, iterations=1):
        result = np.asarray(image)
        for _ in range(iterations):
            result = ndimage.grey_dilation(result, footprint=np.asarray(kernel, dtype=bool), mode="constant", cval=0)
        return result

    pg.snakes.cv2 = pg.snakes2.cv2 = SimpleNamespace(dilate=dilate, useOptimized=lambda: False)
    # Sync clients can transiently hold a scratch directory after publication.
    # Such leftovers are never interpreted as dataset samples.
    with tempfile.TemporaryDirectory(prefix=expected["sample_id"] + "-", dir=work, ignore_cleanup_errors=True) as scratch:
        settings = GenerationSettings(output_root=scratch, **SETTINGS)
        job = SceneJob(int(expected["sample_id"].split("_")[1]), expected["split"],
                       int(expected["betti0"]), int(expected["label"]), int(expected["scene_seed"]))
        row = generate_scene(job, settings)
        row.update(compute_qati_v2_ssl_delta(Path(scratch) / row["image_path"]))
        # Base corpus underwent a solid-metadata V2 migration; extension kept
        # the historical solid bridge column. Neither is the training delta.
        if job.sample_id < 17800:
            from PIL import Image
            with Image.open(Path(scratch) / row["solid_without_points_path"]) as im:
                foreground = np.asarray(im, dtype=np.float32) / 255.0 > 0.1
            _, row["bridge_betti0"], _ = qati_delta_from_foreground(
                foreground, json.loads(row["origin_yx"]), json.loads(row["terminal_yx"]), thickness=3.0)
        if not same_row(row, expected):
            differences = [k for k in expected if not same_row({k: row.get(k)}, {k: expected[k]})]
            raise ValueError(f"Regenerated metadata differs for {expected['sample_id']}: {differences}. Check dependency versions; no files published.")
        hashes = record["pixels_sha256"]
        if hashes is None:
            hashes = {v: pixel_hash(Path(scratch) / expected[v + "_path"]) for v in VIEWS}
        verify_images(Path(scratch), [{**record, "pixels_sha256": hashes}], targets=True)
        for view in VIEWS:
            relative = expected[view + "_path"]
            destination = root / relative
            if destination.exists():
                if pixel_hash(destination) != hashes[view]:
                    raise ValueError(f"Existing image differs: {relative}")
            else:
                write_new(destination, (Path(scratch) / relative).read_bytes())
        if record["pixels_sha256"] is None:
            write_new(root / pixel_record_name(record), json_bytes(pixel_record(record, hashes)))
    return expected["sample_id"]


def run_stage(root, m, digest, stage, *, workers=1, resume=False,
              dry_run=False, verify_only=False, render=render_record):
    root = Path(root)
    for parent in (root, *root.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError("Dataset root/ancestors may not be symlinks or junctions")
    root = root.resolve()
    if stage not in STAGES or workers < 1:
        raise ValueError("Invalid stage/workers")
    completed, managed = inspect_existing(root, m, digest, stage)
    predecessor = STAGES[STAGES.index(stage) - 1] if stage != 4000 else None
    if predecessor is not None and predecessor not in completed:
        raise ValueError(f"First complete stage {predecessor}; expansion does not silently create prerequisites")
    records = selected_records(m, stage)
    unverified = [r for r in records if r["pixels_sha256"] is None]
    if unverified:
        print(f"Note: {len(unverified)} extension scenes have reference metadata but no original pixel hashes. Generated scenes must match the recorded metadata; their pixel hashes will be saved locally. Original-image equivalence is not established.", flush=True)
    # Completed scenes must all exist. Partial next-stage scenes may be missing.
    if completed:
        verify_images(root, selected_records(m, max(completed)))
    missing = verify_images(root, records, allow_missing=True)
    done_ids = {r["row"]["sample_id"] for r in selected_records(m, max(completed))} if completed else set()
    partial = managed and stage not in completed and any(
        (root / r["row"][v + "_path"]).exists()
        for r in records if r["row"]["sample_id"] not in done_ids for v in VIEWS)
    if verify_only:
        if stage not in completed or missing:
            raise ValueError("Stage is incomplete; verify-only never generates data")
        verify_images(root, records, targets=True)
        print(f"Stage {stage}: all metadata, pixels and image-derived V2 targets verified", flush=True)
        return
    if stage in completed:
        print(f"Stage {stage} already complete and verified; nothing changed", flush=True)
        return
    if partial and not resume and not dry_run:
        raise ValueError("Interrupted stage detected; use --resume after inspecting the output")
    print(f"Stage {stage}: {len(records)} total scenes, {len(missing)} scenes need rendering; val/test unchanged", flush=True)
    if dry_run:
        print("Dry run: no files written", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        # Recheck after locking to prevent two stage processes racing.
        latest, _ = inspect_existing(root, m, digest, stage) if managed else ([], False)
        if latest != completed:
            raise ValueError("Dataset state changed during preflight; run again")
        if not managed:
            unexpected = [p for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() != "generation/run.lock"]
            if unexpected:
                raise ValueError("Output changed during preflight; refusing to merge")
        write_new(root / "generation/dataset.json", json_bytes(identity(m, digest)))
        write_new(root / "summary.json", json_bytes(summary(m, digest)))
        tasks = [(str(root), r) for r in missing]
        if workers == 1:
            for index, task in enumerate(tasks, 1):
                render(task)
                if index % 25 == 0 or index == len(tasks):
                    print(f"Rendered {index}/{len(tasks)} scenes", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for index, _ in enumerate(pool.map(render, tasks, chunksize=1), 1):
                    if index % 25 == 0 or index == len(tasks):
                        print(f"Rendered {index}/{len(tasks)} scenes", flush=True)
        # A completion marker is published only after whole-stage verification.
        verify_images(root, records, targets=True)
        files = metadata_files(m, stage)
        for name, content in files.items():
            write_new(root / name, content)
        write_new(root / f"generation/stage_{stage}.complete.json", json_bytes(receipt(m, digest, stage, files, root)))
        print(f"Stage {stage} complete: {len(records)} verified scenes", flush=True)


def main(stage):
    parser = argparse.ArgumentParser(description=f"Build/verify the paper's {stage}-scene training stage (plus fixed 900 val + 900 test).")
    parser.add_argument("--output-root", type=Path, required=True, help="New/managed dataset directory; never the original research corpus")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true", help="Verify existing files, then render missing scenes only")
    parser.add_argument("--dry-run", action="store_true", help="Read-only validation and generation plan")
    parser.add_argument("--verify-only", action="store_true", help="Verify completed stage including recomputed V2 targets; no writes")
    args = parser.parse_args()
    if args.dry_run and args.verify_only:
        parser.error("Choose either --dry-run or --verify-only")
    try:
        m, digest = load_manifest()
        run_stage(args.output_root, m, digest, stage, workers=args.workers, resume=args.resume,
                  dry_run=args.dry_run, verify_only=args.verify_only)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
