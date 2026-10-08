"""Generate one paired scene; split membership belongs to the frozen manifest."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import numpy as np
from PIL import Image
from scipy import ndimage
from data_generation._lib import rendering as pg


DEFAULT_BETTI_VALUES = tuple(range(5, 14))


VIEWS = ("dashed_with_points", "solid_without_points", "solid_with_points", "dashed_without_points")


EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


class SceneGenerationError(RuntimeError):
    """Raised when a scene cannot be accepted within the retry limit."""


@dataclass(frozen=True)
class SceneJob:
    sample_id: int
    split: str
    betti0: int
    connectivity: int
    scene_seed: int


@dataclass(frozen=True)
class GenerationSettings:
    output_root: str
    path_length: int
    window_size: int
    line_thickness: float
    marker_radius: float
    bridge_thickness: float
    component_threshold: float
    max_scene_attempts: int


def save_uint8_png(path: Path, array: np.ndarray) -> None:
    """Save one clipped float image as an 8-bit grayscale PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    uint8_image = np.rint(clipped * 255.0).astype(np.uint8)
    Image.fromarray(uint8_image, mode="L").save(path, format="PNG")


def count_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Label foreground with the canonical 8-neighbourhood."""

    return ndimage.label(np.asarray(binary, dtype=bool), structure=EIGHT_CONNECTED)


def nearest_component_id(
    labels: np.ndarray, coordinate_yx: Sequence[int], radius: int
) -> int | None:
    """Return the nearest non-zero component around one query coordinate."""

    y0, x0 = (int(value) for value in coordinate_yx)
    height, width = labels.shape
    best_id: int | None = None
    best_distance: int | None = None
    for y in range(max(0, y0 - radius), min(height, y0 + radius + 1)):
        for x in range(max(0, x0 - radius), min(width, x0 + radius + 1)):
            component_id = int(labels[y, x])
            if component_id == 0:
                continue
            distance = (y - y0) ** 2 + (x - x0) ** 2
            if best_distance is None or distance < best_distance:
                best_id = component_id
                best_distance = distance
    return best_id


def connectivity_from_labels(
    labels: np.ndarray,
    origin_yx: Sequence[int],
    terminal_yx: Sequence[int],
    search_radius: int,
) -> int:
    """Measure whether both query endpoints belong to the same component."""

    origin_component = nearest_component_id(labels, origin_yx, search_radius)
    terminal_component = nearest_component_id(labels, terminal_yx, search_radius)
    if origin_component is None or terminal_component is None:
        return -1
    return int(origin_component == terminal_component)


def render_virtual_bridge(
    window_size: int,
    origin_yx: Sequence[int],
    terminal_yx: Sequence[int],
    thickness: float,
) -> np.ndarray:
    """Render the unobserved query-to-query bridge used only for supervision."""

    points = [
        [int(origin_yx[0]), int(origin_yx[1])],
        [int(terminal_yx[0]), int(terminal_yx[1])],
    ]
    return pg.draw_solid_polyline(
        [window_size, window_size], points, thickness, 2, 1.0
    )


def render_point_markers(
    window_size: int,
    origin_yx: Sequence[int],
    terminal_yx: Sequence[int],
    radius: float,
    antialias_scale: int = 2,
) -> np.ndarray:
    """Render the two query markers shared by both marked input views."""

    canvas_size = [window_size, window_size]
    return np.maximum(
        pg.draw_marker_circle(canvas_size, origin_yx, radius, antialias_scale),
        pg.draw_marker_circle(canvas_size, terminal_yx, radius, antialias_scale),
    )


def _generator_args(
    settings: GenerationSettings, betti0: int, sample_id: int
) -> SimpleNamespace:
    padding = 10 if settings.window_size == 128 else 22
    seed_distance = 18 if settings.window_size == 128 else 27
    distractor_length = max(2, settings.path_length // 3)
    return SimpleNamespace(
        contour_path=settings.output_root,
        batch_id=sample_id,
        window_size=[settings.window_size, settings.window_size],
        padding=padding,
        antialias_scale=2,
        LABEL=1,
        seed_distance=seed_distance,
        marker_radius=settings.marker_radius,
        contour_length=settings.path_length,
        distractor_length=distractor_length,
        num_distractor_snakes=betti0 - 2,
        snake_contrast_list=[1.0],
        use_single_paddles=False,
        max_target_contour_retrial=10,
        max_distractor_contour_retrial=10,
        max_paddle_retrial=8,
        continuity=1.8,
        paddle_length=5,
        paddle_thickness=settings.line_thickness,
        paddle_margin_list=[3],
        paddle_contrast_list=[1.0],
        pause_display=False,
        save_images=False,
        save_metadata=False,
        marker_clearance=3.0,
        solid_component_threshold=settings.component_threshold,
        marker_component_search_radius=6,
    )


def _normalised_point(point_yx: Sequence[int], size: int) -> list[float]:
    denominator = max(1, size - 1)
    return [float(point_yx[0]) / denominator, float(point_yx[1]) / denominator]


def generate_scene(job: SceneJob, settings: GenerationSettings) -> dict[str, Any]:
    """Generate, validate, and save one deterministic paired scene."""

    random.seed(job.scene_seed)
    np.random.seed(job.scene_seed)
    args = _generator_args(settings, job.betti0, job.sample_id)
    margin = int(args.paddle_margin_list[0])
    marker_exclusion_radius = args.marker_radius + args.marker_clearance

    for attempt in range(1, settings.max_scene_attempts + 1):
        target_images, mask, origin_tips, terminal_tips, target_points = (
            pg.generate_target_snakes_with_points(args, margin)
        )
        contrast = max(args.snake_contrast_list)
        target_solids = [
            pg.draw_solid_polyline(
                args.window_size,
                points,
                args.paddle_thickness,
                args.antialias_scale,
                contrast,
            )
            for points in target_points
        ]

        origin_index = int(np.random.randint(0, 2))
        terminal_index = origin_index if job.connectivity == 1 else 1 - origin_index
        origin_yx = [int(value) for value in origin_tips[origin_index]]
        terminal_yx = [int(value) for value in terminal_tips[terminal_index]]
        marker_specs = [(origin_yx, origin_index), (terminal_yx, terminal_index)]

        empty_distractor = np.zeros_like(target_solids[0])
        if not pg.marker_regions_are_clear(
            target_solids,
            empty_distractor,
            marker_specs,
            marker_exclusion_radius,
            settings.component_threshold,
        ):
            continue

        dashed_curves = np.maximum(target_images[0], target_images[1])
        distractor_mask = pg.reserve_marker_regions(
            mask, [origin_yx, terminal_yx], marker_exclusion_radius
        )
        dashed_curves, _, distractor_points = pg.make_distractor_snakes_with_points(
            dashed_curves, distractor_mask, args, margin
        )
        if len(distractor_points) != args.num_distractor_snakes:
            continue

        distractor_solid = np.zeros_like(target_solids[0])
        for points in distractor_points:
            distractor_solid = np.maximum(
                distractor_solid,
                pg.draw_solid_polyline(
                    args.window_size,
                    points,
                    args.paddle_thickness,
                    args.antialias_scale,
                    contrast,
                ),
            )
        if not pg.marker_regions_are_clear(
            target_solids,
            distractor_solid,
            marker_specs,
            marker_exclusion_radius,
            settings.component_threshold,
        ):
            continue

        solid_without_points = np.maximum(
            np.maximum(target_solids[0], target_solids[1]), distractor_solid
        )
        solid_binary = solid_without_points > settings.component_threshold
        solid_labels, measured_betti0 = count_components(solid_binary)
        if measured_betti0 != job.betti0:
            continue
        measured_connectivity = connectivity_from_labels(
            solid_labels,
            origin_yx,
            terminal_yx,
            args.marker_component_search_radius,
        )
        if measured_connectivity != job.connectivity:
            continue

        markers = render_point_markers(
            settings.window_size,
            origin_yx,
            terminal_yx,
            settings.marker_radius,
            args.antialias_scale,
        )
        dashed_with_points = np.maximum(dashed_curves, markers)
        solid_with_points = np.maximum(solid_without_points, markers)

        bridge = render_virtual_bridge(
            settings.window_size,
            origin_yx,
            terminal_yx,
            settings.bridge_thickness,
        )
        _, bridge_betti0 = count_components(
            solid_binary | (bridge > settings.component_threshold)
        )
        delta_betti0 = measured_betti0 - bridge_betti0
        if delta_betti0 < 0:
            continue
        if job.connectivity == 0 and delta_betti0 < 1:
            continue

        filename = f"sample_{job.sample_id:06d}.png"
        output_root = Path(settings.output_root)
        dashed_relative = Path(job.split) / "dashed_with_points" / filename
        solid_relative = Path(job.split) / "solid_without_points" / filename
        solid_marked_relative = Path(job.split) / "solid_with_points" / filename
        dashed_unmarked_relative = Path(job.split) / "dashed_without_points" / filename
        save_uint8_png(output_root / dashed_relative, dashed_with_points)
        save_uint8_png(output_root / solid_relative, solid_without_points)
        save_uint8_png(output_root / solid_marked_relative, solid_with_points)
        save_uint8_png(output_root / dashed_unmarked_relative, dashed_curves)

        origin_normalised = _normalised_point(origin_yx, settings.window_size)
        terminal_normalised = _normalised_point(terminal_yx, settings.window_size)
        return {
            "sample_id": f"sample_{job.sample_id:06d}",
            "split": job.split,
            "image_path": dashed_relative.as_posix(),
            "dashed_with_points_path": dashed_relative.as_posix(),
            "solid_without_points_path": solid_relative.as_posix(),
            "solid_with_points_path": solid_marked_relative.as_posix(),
            "dashed_without_points_path": dashed_unmarked_relative.as_posix(),
            "label": job.connectivity,
            "betti0": measured_betti0,
            "betti_0": measured_betti0,
            "betti0_class": DEFAULT_BETTI_VALUES.index(measured_betti0),
            "bridge_betti0": bridge_betti0,
            "delta_betti0": delta_betti0,
            "origin_yx": json.dumps(origin_yx, separators=(",", ":")),
            "terminal_yx": json.dumps(terminal_yx, separators=(",", ":")),
            "origin_yx_normalised": json.dumps(
                origin_normalised, separators=(",", ":")
            ),
            "terminal_yx_normalised": json.dumps(
                terminal_normalised, separators=(",", ":")
            ),
            "path_length": settings.path_length,
            "distractor_paths": args.num_distractor_snakes,
            "total_paths": measured_betti0,
            "scene_seed": job.scene_seed,
            "attempts": attempt,
        }

    raise SceneGenerationError(
        f"Could not generate sample {job.sample_id} after "
        f"{settings.max_scene_attempts} attempts."
    )
