"""Versioned construction of query-anchored topological intervention targets."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage


QATI_V2_TARGET_VERSION = "analytic_capsule_pixel_centres_r1p5_v2"
EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


def analytic_capsule_mask(
    shape: tuple[int, int],
    point_1_yx: Sequence[float],
    point_2_yx: Sequence[float],
    thickness: float = 3.0,
) -> np.ndarray:
    """Return a native-resolution, endpoint-symmetric binary segment capsule.

    A pixel belongs to the bridge when its centre is at most ``thickness / 2``
    from the closed Euclidean line segment joining the two query points.  No
    supersampling, image resizing, greyscale rasterisation, or thresholding is
    involved.
    """

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"Expected a positive 2-D shape, got {shape!r}")
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError(f"Thickness must be positive and finite, got {thickness!r}")

    point_1 = np.asarray(point_1_yx, dtype=np.float64)
    point_2 = np.asarray(point_2_yx, dtype=np.float64)
    if point_1.shape != (2,) or point_2.shape != (2,):
        raise ValueError("Query points must each have shape (2,) in [y, x] order")
    if not np.isfinite(point_1).all() or not np.isfinite(point_2).all():
        raise ValueError("Query points must contain finite coordinates")

    yy, xx = np.ogrid[: shape[0], : shape[1]]
    direction = point_2 - point_1
    denominator = float(direction @ direction)
    if denominator == 0.0:
        distance_squared = (yy - point_1[0]) ** 2 + (xx - point_1[1]) ** 2
    else:
        projection = (
            (yy - point_1[0]) * direction[0]
            + (xx - point_1[1]) * direction[1]
        ) / denominator
        projection = np.clip(projection, 0.0, 1.0)
        nearest_y = point_1[0] + projection * direction[0]
        nearest_x = point_1[1] + projection * direction[1]
        distance_squared = (yy - nearest_y) ** 2 + (xx - nearest_x) ** 2

    radius_squared = (thickness / 2.0) ** 2
    return np.asarray(distance_squared <= radius_squared + 1e-12, dtype=bool)


def count_components(mask: np.ndarray) -> int:
    """Count foreground components using the canonical 8-connectivity."""

    _, count = ndimage.label(np.asarray(mask, dtype=bool), structure=EIGHT_CONNECTED)
    return int(count)


def qati_delta_from_foreground(
    foreground: np.ndarray,
    point_1_yx: Sequence[float],
    point_2_yx: Sequence[float],
    thickness: float = 3.0,
) -> tuple[int, int, np.ndarray]:
    """Return ``(initial_betti0, bridged_betti0, bridge_mask)`` for QATI V2."""

    foreground = np.asarray(foreground, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError(f"Expected a 2-D foreground mask, got {foreground.shape!r}")
    bridge = analytic_capsule_mask(
        foreground.shape, point_1_yx, point_2_yx, thickness=thickness
    )
    initial_betti0 = count_components(foreground)
    bridged_betti0 = count_components(foreground | bridge)
    return initial_betti0, bridged_betti0, bridge
