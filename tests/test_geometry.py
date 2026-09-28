"""Geometry helper tests (SPEC.md §11, Faz 3 prerequisite)."""

from __future__ import annotations

import cv2
import numpy as np

from counterpart.score.geometry import (
    boundary_points,
    curvature_energy,
    dilate,
    distance_transforms,
    rings,
    signed_distance,
)


def _mask(size: int = 128) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    cv2.circle(mask, (size // 2, size // 2), 30, 1, -1)
    return mask.astype(bool)


def test_distance_transforms() -> None:
    mask = _mask()
    dist_in, dist_out = distance_transforms(mask)
    assert dist_in[64, 64] == np.max(dist_in)
    assert dist_in[64, 64] > 25
    assert dist_out[64, 64] == 0.0
    assert dist_out[5, 5] > 30
    sdf = signed_distance(dist_in, dist_out)
    assert sdf[64, 64] < 0 < sdf[5, 5]


def test_rings_and_visible() -> None:
    damage = np.zeros((128, 128), dtype=bool)
    damage[50:70, 50:70] = True
    obj = _mask()
    ring_in, ring_out, visible = rings(damage, obj, width=8)
    assert ring_in.sum() > 0 and ring_out.sum() > 0
    assert not (ring_in & ~damage).any()
    assert not (ring_out & damage).any()
    assert not (visible & damage).any()
    assert not (visible & dilate(damage, 8)).any(), "V must exclude the dilated damage mask"


def test_boundary_points_on_mask_edge() -> None:
    mask = _mask()
    points = boundary_points(mask, stride=4)
    assert len(points) > 10
    assert points[:, 0].min() >= 0 and points[:, 0].max() < 128
    # every boundary point must sit near the mask contour: its distance to the mask is small
    for x, y in points[:20]:
        iy, ix = int(y), int(x)
        neighbourhood = mask[max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3]
        assert neighbourhood.any()


def test_curvature_energy_straight_vs_zigzag() -> None:
    straight = np.stack([np.arange(50, dtype=float), np.zeros(50)], axis=1)
    zigzag = straight.copy()
    zigzag[::2, 1] = 3.0
    assert curvature_energy(straight) < curvature_energy(zigzag)
