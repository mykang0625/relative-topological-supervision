"""Only the rendering operations used by the frozen paper scenes."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def install_cv2_fallback() -> None:
    """Provide the two OpenCV operations used by the legacy generator."""
    try:
        import cv2  # noqa: F401
        return
    except (ImportError, OSError):
        sys.modules.pop("cv2", None)

    fallback = ModuleType("cv2")
    fallback.useOptimized = lambda: False

    def dilate(image, kernel, iterations=1):
        result = np.asarray(image)
        footprint = np.asarray(kernel, dtype=bool)
        for _ in range(iterations):
            result = ndimage.grey_dilation(
                result, footprint=footprint, mode="constant", cval=0
            )
        return result

    fallback.dilate = dilate
    sys.modules["cv2"] = fallback


def sample_contrast(values: list[float]) -> float:
    if not values:
        return 1.0
    return values[np.random.randint(0, len(values))]


def generate_target_snakes_with_points(args: SimpleNamespace, margin: int):
    """Generate the two main snakes and retain their ordered polyline vertices."""
    small_struct = snakes.generate_dilation_struct(margin)
    large_struct = snakes.generate_dilation_struct(margin * args.antialias_scale)

    while True:
        snake_contrasts = args.snake_contrast_list * 2
        random.shuffle(snake_contrasts)
        snake_contrasts = snake_contrasts[:2]
        result = snakes2.initialize_two_seeds(
            args.window_size,
            args.padding,
            args.seed_distance,
            args.paddle_length,
            args.paddle_thickness,
            margin,
            snake_contrasts,
            args.paddle_contrast_list,
            small_struct,
            large_struct,
            args.max_paddle_retrial,
            args.antialias_scale,
            display=False,
        )
        images, mask, segment_masks, pivots, orientations, origin_tips, success = result
        if not success:
            continue

        points = []
        for index in range(2):
            first_head = snakes.translate_coord(
                pivots[index], orientations[index], args.paddle_length + margin
            )
            points.append([origin_tips[index], first_head])

        terminal_tips = [[0, 0], [0, 0]]
        failed = False
        for _ in range(args.contour_length - 1):
            contrast = sample_contrast(args.paddle_contrast_list)
            for index in range(2):
                result = snakes.extend_snake(
                    list(pivots[index]),
                    orientations[index],
                    segment_masks[index],
                    images[index],
                    mask,
                    args.max_paddle_retrial,
                    args.paddle_length,
                    args.paddle_thickness,
                    margin,
                    args.continuity,
                    contrast * snake_contrasts[index],
                    small_struct,
                    large_struct,
                    aa_scale=args.antialias_scale,
                    display=False,
                    forced_current_pivot=None,
                )
                (
                    images[index],
                    mask,
                    segment_masks[index],
                    pivots[index],
                    orientations[index],
                    terminal_tips[index],
                    success,
                ) = result
                if not success:
                    failed = True
                    break
                points[index].append(terminal_tips[index])
            if failed:
                break
        if failed:
            continue

        mask = np.maximum(mask, segment_masks[-1])
        return images, mask, origin_tips, terminal_tips, points


def seed_snake_with_points(
    image: np.ndarray,
    mask: np.ndarray,
    args: SimpleNamespace,
    margin: int,
    small_struct: np.ndarray,
    large_struct: np.ndarray,
):
    contrast = sample_contrast(args.snake_contrast_list)
    result = snakes.seed_snake(
        image,
        mask,
        args.max_paddle_retrial,
        args.paddle_length,
        args.paddle_thickness,
        margin,
        contrast,
        small_struct,
        large_struct,
        aa_scale=args.antialias_scale,
        display=False,
        stop_with_availability=0.01,
    )
    image, mask, segment_mask, pivot, orientation, success = result
    if not success:
        return image, mask, segment_mask, pivot, orientation, None, False
    tail = snakes.translate_coord(pivot, orientation, margin)
    head = snakes.translate_coord(pivot, orientation, args.paddle_length + margin)
    return image, mask, segment_mask, pivot, orientation, [tail, head], True


def make_snake_with_points(
    image: np.ndarray,
    mask: np.ndarray,
    args: SimpleNamespace,
    margin: int,
    num_segments: int,
):
    original_image, original_mask = image, mask
    small_struct = snakes.generate_dilation_struct(margin)
    large_struct = snakes.generate_dilation_struct(margin * args.antialias_scale)
    result = seed_snake_with_points(image.copy(), mask.copy(), args, margin, small_struct, large_struct)
    image, mask, segment_mask, pivot, orientation, points, success = result
    if not success:
        return original_image, original_mask, None, False

    for _ in range(num_segments - 1):
        result = snakes.extend_snake(
            list(pivot),
            orientation,
            segment_mask,
            image,
            mask,
            args.max_paddle_retrial,
            args.paddle_length,
            args.paddle_thickness,
            margin,
            args.continuity,
            sample_contrast(args.snake_contrast_list),
            small_struct,
            large_struct,
            aa_scale=args.antialias_scale,
            display=False,
            forced_current_pivot=None,
        )
        image, mask, segment_mask, pivot, orientation, terminal_tip, success = result
        if not success:
            return original_image, original_mask, None, False
        points.append(terminal_tip)

    mask = np.maximum(mask, segment_mask)
    return image, mask, points, True


def make_distractor_snakes_with_points(
    image: np.ndarray,
    mask: np.ndarray,
    args: SimpleNamespace,
    margin: int,
):
    all_points = []
    for _ in range(args.num_distractor_snakes):
        for _ in range(args.max_distractor_contour_retrial + 1):
            next_image, next_mask, points, success = make_snake_with_points(
                image, mask, args, margin, args.distractor_length
            )
            if success:
                image, mask = next_image, next_mask
                all_points.append(points)
                break
    return image, mask, all_points


def draw_solid_polyline(
    window_size: list[int],
    points_yx: list[list[int]],
    thickness: float,
    aa_scale: int,
    contrast: float,
) -> np.ndarray:
    """Render ordered [y, x] vertices as one anti-aliased solid path."""
    height, width = window_size
    image = Image.new("L", (width * aa_scale, height * aa_scale), 0)
    points_xy = [
        (int(round(x * aa_scale)), int(round(y * aa_scale))) for y, x in points_yx
    ]
    ImageDraw.Draw(image).line(
        points_xy,
        fill=255,
        width=max(1, int(round(thickness * aa_scale))),
    )
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.clip(np.asarray(image, dtype=np.float32) / 255.0 * contrast, 0.0, 1.0)


def draw_marker_circle(
    window_size: list[int], coordinate_yx: list[int], radius: float, aa_scale: int
) -> np.ndarray:
    """Draw a float marker circle with the paired renderer's anti-aliasing."""
    height, width = window_size
    image = np.zeros((height * aa_scale, width * aa_scale), dtype=np.float32)
    y, x = np.ogrid[
        -coordinate_yx[0] * aa_scale : (height - coordinate_yx[0]) * aa_scale,
        -coordinate_yx[1] * aa_scale : (width - coordinate_yx[1]) * aa_scale,
    ]
    image[x * x + y * y <= (radius * aa_scale) ** 2] = 1.0
    resized = Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32)


def circular_region(
    window_size: list[int], coordinate_yx: list[int], radius: float
) -> np.ndarray:
    """Return a non-antialiased circular boolean mask around one [y, x] point."""

    height, width = window_size
    y0, x0 = (float(value) for value in coordinate_yx)
    y, x = np.ogrid[:height, :width]
    return (x - x0) ** 2 + (y - y0) ** 2 <= radius**2


def reserve_marker_regions(
    mask: np.ndarray,
    marker_coordinates_yx: list[list[int]],
    radius: float,
) -> np.ndarray:
    """Prevent subsequently generated distractors from entering marker regions."""

    reserved = mask.copy()
    for coordinate_yx in marker_coordinates_yx:
        reserved[circular_region(list(mask.shape), coordinate_yx, radius)] = 1.0
    return reserved


def marker_regions_are_clear(
    target_solids: list[np.ndarray],
    distractor_solid: np.ndarray,
    marker_specs: list[tuple[list[int], int]],
    radius: float,
    threshold: float,
) -> bool:
    """Check that each marker region contains only its selected target path."""

    if radius <= 0:
        return True
    for coordinate_yx, own_target_index in marker_specs:
        foreign_solid = np.maximum(
            target_solids[1 - own_target_index], distractor_solid
        )
        region = circular_region(
            list(foreign_solid.shape), coordinate_yx, radius
        )
        if np.any((foreign_solid > threshold) & region):
            return False
    return True


PATHFINDER_CODE = Path(__file__).resolve().parents[1] / "vendor"
sys.path.insert(0, str(PATHFINDER_CODE))
install_cv2_fallback()
import snakes
import snakes2
