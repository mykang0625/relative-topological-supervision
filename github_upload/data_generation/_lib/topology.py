"""Digital-topology primitives used by the RETA QATI pilot.

Foreground vessels use 8-connectivity and the complementary background uses
4-connectivity.  This dual convention avoids the usual digital-topology paradox in
which a diagonal contact is simultaneously connected in foreground and background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi


FOREGROUND_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
BACKGROUND_STRUCTURE = ndi.generate_binary_structure(2, 1).astype(np.uint8)


@dataclass(frozen=True)
class BettiNumbers:
    beta0: int
    beta1: int


@dataclass(frozen=True)
class BridgeIntervention:
    before: BettiNumbers
    after: BettiNumbers

    @property
    def delta_beta0(self) -> int:
        """Components removed by adding the bridge: beta0(before)-beta0(after)."""

        return self.before.beta0 - self.after.beta0

    @property
    def delta_beta1(self) -> int:
        """Signed loop change under bridge addition: beta1(after)-beta1(before)."""

        return self.after.beta1 - self.before.beta1


def label_foreground(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label an arbitrary binary foreground mask under 8-connectivity."""

    binary = np.asarray(mask, dtype=bool)
    return ndi.label(binary, structure=FOREGROUND_STRUCTURE)


def betti_numbers(mask: np.ndarray) -> BettiNumbers:
    """Compute planar beta0 and beta1 for a binary raster.

    beta1 is the number of 4-connected background components that do not touch the
    canvas boundary.  The unbounded exterior background component is therefore not a
    hole.  This definition is exact for the foreground-8/background-4 convention.
    """

    binary = np.asarray(mask, dtype=bool)
    _, beta0 = label_foreground(binary)
    background_labels, background_count = ndi.label(
        ~binary, structure=BACKGROUND_STRUCTURE
    )
    border_labels = set(
        np.unique(
            np.concatenate(
                (
                    background_labels[0, :],
                    background_labels[-1, :],
                    background_labels[:, 0],
                    background_labels[:, -1],
                )
            )
        ).tolist()
    )
    beta1 = sum(
        component_id not in border_labels
        for component_id in range(1, background_count + 1)
    )
    return BettiNumbers(beta0=int(beta0), beta1=int(beta1))


def analytic_capsule_mask(
    shape: Sequence[int],
    p1_yx: Sequence[float],
    p2_yx: Sequence[float],
    radius: float,
) -> np.ndarray:
    """Rasterise a closed line-segment capsule using native pixel centres."""

    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D shape, got {tuple(shape)}")
    if radius <= 0:
        raise ValueError("radius must be positive")

    height, width = int(shape[0]), int(shape[1])
    p1 = np.asarray(p1_yx, dtype=np.float64)
    p2 = np.asarray(p2_yx, dtype=np.float64)
    if p1.shape != (2,) or p2.shape != (2,):
        raise ValueError("p1_yx and p2_yx must each contain (y, x)")

    pad = float(radius) + 1.0
    y0 = max(0, int(np.floor(min(p1[0], p2[0]) - pad)))
    y1 = min(height, int(np.ceil(max(p1[0], p2[0]) + pad)) + 1)
    x0 = max(0, int(np.floor(min(p1[1], p2[1]) - pad)))
    x1 = min(width, int(np.ceil(max(p1[1], p2[1]) + pad)) + 1)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    points = np.stack((yy, xx), axis=-1).astype(np.float64)
    direction = p2 - p1
    squared_length = float(np.dot(direction, direction))
    if squared_length == 0.0:
        squared_distance = np.sum((points - p1) ** 2, axis=-1)
    else:
        projection = np.sum((points - p1) * direction, axis=-1) / squared_length
        projection = np.clip(projection, 0.0, 1.0)
        closest = p1 + projection[..., None] * direction
        squared_distance = np.sum((points - closest) ** 2, axis=-1)

    result = np.zeros((height, width), dtype=bool)
    result[y0:y1, x0:x1] = squared_distance <= float(radius) ** 2
    return result


def bridge_intervention(
    mask: np.ndarray,
    p1_yx: Sequence[float],
    p2_yx: Sequence[float],
    radius: float,
    *,
    before: BettiNumbers | None = None,
) -> BridgeIntervention:
    """Compute beta0/beta1 before and after adding an analytic capsule."""

    binary = np.asarray(mask, dtype=bool)
    before_numbers = before if before is not None else betti_numbers(binary)
    bridge = analytic_capsule_mask(binary.shape, p1_yx, p2_yx, radius)
    after_numbers = betti_numbers(binary | bridge)
    result = BridgeIntervention(before=before_numbers, after=after_numbers)
    if result.delta_beta0 < 0:
        raise AssertionError("Adding foreground unexpectedly increased beta0")
    return result


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
