"""Replay the frozen dense-clutter corpus with its original target convention."""
from __future__ import annotations

from data_generation._lib.rendering import snakes, snakes2
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Tuple
from data_generation._lib.scenes import count_components, render_virtual_bridge
from concurrent.futures import ProcessPoolExecutor
import tempfile
from data_generation._lib import common as pc


def save_uint8(path: str | Path, arr: np.ndarray) -> None:
    out = np.nan_to_num(np.asarray(arr))
    scale = 255.0 if out.size and np.issubdtype(out.dtype, np.floating) and float(out.max()) <= 2.0 else 1.0
    if out.dtype != np.uint8 or scale != 1.0:
        out = np.clip(out * scale, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


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


def generate_single_scene(
    sample_id: int,
    split: str,
    label: int,
    seed: int,
    output_dir: Path,
    padding: int = 10,
    bridge_thickness: float = 3.0,
    *,
    start_attempt: int = 0,
    attempt_limit: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)

    window_size = [128, 128]
    seed_distance = 20
    contour_length = 14
    paddle_length = 5
    paddle_thickness = 1.5
    continuity = 1.8
    marker_radius = 3.0
    antialias_scale = 2
    distractor_budget = 35
    distractor_length = 4
    num_distractor_snakes = distractor_budget // distractor_length  # 8
    snake_contrast_list = [0.9]
    paddle_contrast_list = [1.0]
    paddle_margin_list = [2, 3]

    max_paddle_retrial = 8
    max_distractor_contour_retrial = 10

    max_attempts = 50000
    if not 0 <= start_attempt < max_attempts or (attempt_limit is not None and attempt_limit < 1):
        raise ValueError("Invalid reference-attempt bounds")
    # Each attempt is independent after its seed is drawn. Replaying a frozen
    # accepted attempt need not simulate all earlier rejected geometries.
    for _ in range(start_attempt):
        rng.randint(1, 10**9)
    stop_attempt = max_attempts if attempt_limit is None else min(max_attempts, start_attempt + attempt_limit)
    for attempt in range(start_attempt, stop_attempt):
        s = rng.randint(1, 10**9)
        random.seed(s)
        np.random.seed(s)

        margin = random.choice(paddle_margin_list)
        base_num_paddles = 150
        num_paddles_factor = 1.0 / ((7.5 + 13 * margin + 4 * margin * margin) / 123.5)
        total_num_paddles = int(base_num_paddles * num_paddles_factor)

        small_dilation_structs = snakes.generate_dilation_struct(margin)
        large_dilation_structs = snakes.generate_dilation_struct(margin * antialias_scale)

        # Generate two target snakes
        twosnakes, mask, origin_tips, terminal_tips, success = snakes2.two_snakes(
            window_size,
            padding,
            seed_distance,
            contour_length,
            paddle_length,
            paddle_thickness,
            margin,
            continuity,
            small_dilation_structs,
            large_dilation_structs,
            snake_contrast_list,
            paddle_contrast_list,
            max_paddle_retrial,
            antialias_scale,
            display_snake=False,
            display_segment=False,
            allow_shorter_snakes=False,
        )
        if not success:
            continue

        image = np.maximum(twosnakes[0], twosnakes[1])

        # Distractor snakes
        if num_distractor_snakes > 0:
            image, mask = snakes.make_many_snakes(
                image,
                mask,
                num_distractor_snakes,
                max_distractor_contour_retrial,
                distractor_length,
                paddle_length,
                paddle_thickness,
                margin,
                continuity,
                snake_contrast_list,
                max_paddle_retrial,
                antialias_scale,
                display_final=False,
                display_snake=False,
                display_segment=False,
                allow_incomplete=True,
                allow_shorter_snakes=False,
                stop_with_availability=0.01,
            )
            if image is None:
                continue

        # Single paddle distractors
        num_single_paddles = total_num_paddles - 2 * contour_length - num_distractor_snakes * distractor_length
        if num_single_paddles > 0:
            image, _ = snakes.make_many_snakes(
                image,
                mask,
                num_single_paddles,
                max_paddle_retrial,
                1,
                paddle_length,
                paddle_thickness,
                margin,
                continuity,
                snake_contrast_list,
                max_paddle_retrial,
                antialias_scale,
                display_final=False,
                display_snake=False,
                display_segment=False,
                allow_incomplete=True,
                allow_shorter_snakes=False,
                stop_with_availability=0.01,
            )
            if image is None:
                continue

        # Add markers
        origin_mark_idx = random.randint(0, 1)
        terminal_mark_idx = origin_mark_idx if label == 1 else (1 - origin_mark_idx)
        origin_coord = origin_tips[origin_mark_idx]
        terminal_coord = terminal_tips[terminal_mark_idx]

        origin_circle = snakes2.draw_circle(window_size, origin_coord, marker_radius, antialias_scale)
        terminal_circle = snakes2.draw_circle(window_size, terminal_coord, marker_radius, antialias_scale)
        final_image = np.maximum(image, np.maximum(origin_circle, terminal_circle))

        # Save image
        rel_img_dir = Path("images") / split
        abs_img_dir = output_dir / rel_img_dir
        abs_img_dir.mkdir(parents=True, exist_ok=True)
        img_name = f"{split}_{sample_id:05d}_label{label}.png"
        img_path = abs_img_dir / img_name
        save_uint8(img_path, final_image)

        # Precompute Betti-0 and Delta-Betti-0
        # 1. Ground-Truth (GT) bridge using exact endpoint coordinates
        final_norm = final_image.astype(np.float32)
        if final_norm.max() > 2.0:
            final_norm /= 255.0
        binary_before = final_norm > 0.1
        _, b0_initial = count_components(binary_before)

        bridge_gt = render_virtual_bridge(128, origin_coord, terminal_coord, bridge_thickness)
        binary_bridged_gt = binary_before | (bridge_gt > 0.1)
        _, b0_bridged_gt = count_components(binary_bridged_gt)
        delta_betti0_gt = max(0, int(b0_initial - b0_bridged_gt))

        # 2. SSL bridge using template-detected marker coordinates from the image raster
        p1_ssl, p2_ssl = detect_query_markers_ssl(final_norm, threshold=0.1)
        bridge_ssl = render_virtual_bridge(128, p1_ssl, p2_ssl, bridge_thickness)
        binary_bridged_ssl = binary_before | (bridge_ssl > 0.1)
        _, b0_bridged_ssl = count_components(binary_bridged_ssl)
        delta_betti0_ssl = max(0, int(b0_initial - b0_bridged_ssl))

        return {
            "sample_id": sample_id,
            "split": split,
            "label": label,
            "image_path": str(rel_img_dir / img_name),
            "origin_yx": json.dumps([int(origin_coord[0]), int(origin_coord[1])]),
            "terminal_yx": json.dumps([int(terminal_coord[0]), int(terminal_coord[1])]),
            "betti0": int(b0_initial),
            "betti0_bridged": int(b0_bridged_gt),
            "delta_betti0": int(delta_betti0_gt),
            "p1_ssl_yx": json.dumps([float(p1_ssl[0]), float(p1_ssl[1])]),
            "p2_ssl_yx": json.dumps([float(p2_ssl[0]), float(p2_ssl[1])]),
            "betti0_initial_ssl": int(b0_initial),
            "betti0_bridged_ssl": int(b0_bridged_ssl),
            "delta_betti0_ssl": int(delta_betti0_ssl),
            "attempts": attempt + 1,
        }

    raise RuntimeError(f"Failed to generate scene {sample_id} within attempt range {start_attempt + 1}..{stop_attempt}")


BACKEND = "opencv"


def select_backend(name=BACKEND):
    if name == "opencv":
        import cv2
        if not hasattr(cv2, "__version__"):
            raise RuntimeError("Dense-clutter reproduction requires real OpenCV, not the legacy fallback. Install the additional-data requirements.")
        snakes.cv2 = snakes2.cv2 = cv2
    else:
        from types import SimpleNamespace
        import numpy as np
        from scipy import ndimage

        def dilate(image, kernel, iterations=1):
            result = np.asarray(image)
            for _ in range(iterations):
                result = ndimage.grey_dilation(result, footprint=np.asarray(kernel, dtype=bool), mode="constant", cval=0)
            return result

        snakes.cv2 = snakes2.cv2 = SimpleNamespace(dilate=dilate, useOptimized=lambda: False)


def render_record(task):
    root_name, record = task
    root, expected = Path(root_name), record["row"]
    select_backend()
    work = root / "generation/work"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dense-", dir=work, ignore_cleanup_errors=True) as scratch:
        row = generate_single_scene(int(expected["sample_id"]), expected["split"],
                    int(expected["label"]), record["seed"], Path(scratch), 10, 3.0,
                    start_attempt=int(expected["attempts"]) - 1, attempt_limit=1)
        pc.compare_row(row, expected, path_keys=("image_path",))
        image_path = Path(scratch) / expected["image_path"]
        if pc.image_record(image_path) != record["image"]:
            raise ValueError(f"Dense reference pixels differ: {expected['sample_id']}")
        dest = root / expected["image_path"]
        if dest.exists():
            if pc.image_record(dest) != record["image"]:
                raise ValueError(f"Existing dense image differs: {expected['sample_id']}")
        else:
            pc.write_new(dest, image_path.read_bytes())
    return expected["sample_id"]


def metadata(m):
    by_id = {r["row"]["sample_id"]: r["row"] for r in m["records"]}
    return {f"metadata/{name}.csv": pc.csv_bytes([by_id[s] for s in ids], m["columns"])
            for name, ids in m["splits"].items()}


def verify_images(root, records, *, required=False):
    missing = []
    for r in records:
        path = root / r["row"]["image_path"]
        if not path.is_file():
            if required:
                raise ValueError(f"Missing completed image: {path}")
            missing.append(r)
        elif pc.image_record(path) != r["image"]:
            raise ValueError(f"Changed dense-clutter image: {path}")
    return missing


def prepare(root, *, workers=4, resume=False, dry_run=False, verify_only=False):
    root = pc.checked_root(root)
    m, digest = pc.load_reference("dense_clutter_reference")
    ident = pc.json_bytes({"dataset_id": m["dataset_id"], "reference_sha256": digest,
                          "target_version": m["target_version"], "render_backend": BACKEND})
    files = metadata(m)
    files["summary.json"] = pc.json_bytes({"dataset_id": m["dataset_id"], "target_version": m["target_version"],
           "split_sizes": {"train": 16000, "val": 900, "test": 900}, "render_backend": BACKEND,
           "note": "Historical dense-clutter targets, not canonical analytic V2. Accepted attempts replayed from frozen records."})
    marker = "generation/dense.complete.json"
    complete = pc.json_bytes({"reference_sha256": digest, "scenes": len(m["records"]),
                            "files_sha256": {p: pc.sha256(b) for p, b in files.items()}})
    allowed = set(files) | {r["row"]["image_path"] for r in m["records"]} | {marker, "generation/dense.identity.json", "generation/run.lock"}
    identity_path = root / "generation/dense.identity.json"
    if root.exists() and any(root.iterdir()):
        if not identity_path.is_file() or identity_path.read_bytes() != ident:
            raise ValueError("Use a NEW empty output directory or a matching managed dense-clutter directory")
    pc.inspect_files(root, allowed, extra_prefixes=("generation/work/",))
    is_complete = (root / marker).is_file()
    if is_complete and (root / marker).read_bytes() != complete:
        raise ValueError("Invalid dense completion marker")
    pc.verify_payload(root, files, required=is_complete)
    missing = verify_images(root, m["records"], required=is_complete)
    if verify_only and not is_complete:
        raise ValueError("Dense-clutter dataset is incomplete")
    if is_complete:
        print("Dense-clutter corpus verified; no files changed", flush=True)
        return
    if identity_path.exists() and not resume and not dry_run:
        raise ValueError("Interrupted preparation; use --resume")
    print(f"Dense-clutter: {len(missing)} of {len(m['records'])} scenes need generation", flush=True)
    if dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    with pc.exclusive_run(root):
        pc.inspect_files(root, allowed, extra_prefixes=("generation/work/",))
        pc.write_new(identity_path, ident)
        if workers == 1:
            for i, record in enumerate(missing, 1):
                render_record((str(root), record))
                if i % 50 == 0 or i == len(missing):
                    print(f"Dense generated {i}/{len(missing)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, _ in enumerate(pool.map(render_record, [(str(root), r) for r in missing]), 1):
                    if i % 50 == 0 or i == len(missing):
                        print(f"Dense generated {i}/{len(missing)}", flush=True)
        verify_images(root, m["records"], required=True)
        for path, data in files.items():
            pc.write_new(root / path, data)
        pc.write_new(root / marker, complete)
    print("Dense-clutter preparation complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.dry_run and args.verify_only):
        parser.error("Positive workers and at most one inspection mode are required")
    try:
        options = vars(args)
        root = options.pop("output_root")
        prepare(root, **options)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
