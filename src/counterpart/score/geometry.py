"""Shared geometry for score terms (SPEC.md §5.4): distance transforms, rings, boundary."""

from __future__ import annotations

import cv2
import numpy as np


def ellipse_kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), ellipse_kernel(radius)).astype(bool)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return cv2.erode(mask.astype(np.uint8), ellipse_kernel(radius)).astype(bool)


def distance_transforms(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance (px) to the mask boundary inside and outside the mask."""
    m = mask.astype(np.uint8)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return dist_in, dist_out


def signed_distance(dist_in: np.ndarray, dist_out: np.ndarray) -> np.ndarray:
    """Signed distance to ∂M: positive outside, negative inside."""
    return dist_out - dist_in


def rings(
    damage_mask: np.ndarray, object_mask: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R_in, R_out (width ``width``) and the visible object region V (SPEC.md §5.4)."""
    dil = dilate(damage_mask, width)
    ero = erode(damage_mask, width)
    ring_in = damage_mask & ~ero
    ring_out = dil & ~damage_mask
    visible = object_mask & ~dil
    return ring_in, ring_out, visible


def boundary_points(mask: np.ndarray, stride: int = 1) -> np.ndarray:
    """Sample points on ∂M (all external contours, every ``stride``-th pixel). Nx2 (x, y)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("mask has no contours")
    points = np.concatenate([c.reshape(-1, 2) for c in contours if len(c) > 3])
    step = max(1, int(stride))
    return points[::step].astype(np.float64)


def fit_line_direction(points: np.ndarray) -> np.ndarray:
    """Principal direction of a point set (PCA), unit norm."""
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    return direction / max(float(np.linalg.norm(direction)), 1e-9)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in [0, pi/2] between two unoriented directions."""
    c = float(
        np.clip(abs(np.dot(u, v)) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-9), 0, 1)
    )
    return float(np.arccos(c))


def ordered_contour_points(mask: np.ndarray) -> np.ndarray:
    """Largest contour of ``mask`` as an ordered Nx2 float array (for curvature fits)."""
    contours = ordered_contours(mask)
    if not contours:
        raise ValueError("mask has no contours")
    return max(contours, key=lambda pts: len(pts))


def ordered_contours(mask: np.ndarray) -> list[np.ndarray]:
    """All external contours of ``mask`` as ordered Nx2 (x, y) float arrays."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result: list[np.ndarray] = []
    for contour in contours:
        pts = contour.reshape(-1, 2)
        if len(pts) >= 3:
            result.append(pts.astype(np.float64))
    return result


def boolean_runs(flags: np.ndarray, wrap: bool = True) -> list[tuple[int, int]]:
    """Contiguous True runs in ``flags`` as (start, end) index pairs (inclusive)."""
    n = len(flags)
    if n == 0:
        return []
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if flags[i]:
            start = i
            while i < n and flags[i]:
                i += 1
            runs.append((start, i - 1))
        else:
            i += 1
    if wrap and len(runs) > 1 and flags[0] and flags[-1]:
        last, first = runs[-1], runs[0]
        runs = runs[1:-1] + [(last[0], first[1] + n)]
    return runs


def valid_patch_positions(region: np.ndarray, patch: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners of ``patch``-sized windows fully contained in ``region``."""
    integral = cv2.integral(region.astype(np.uint8))
    positions: list[tuple[int, int]] = []
    h, w = region.shape
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            area = (
                integral[y + patch, x + patch]
                - integral[y, x + patch]
                - integral[y + patch, x]
                + integral[y, x]
            )
            if area == patch * patch:
                positions.append((x, y))
    return positions


def curvature_energy(points: np.ndarray, smooth: int = 5) -> float:
    """Mean squared derivative of the tangent angle along an ordered polyline.

    Points are smoothed with a moving average first (index-space), so pixel noise on
    the contour does not dominate.
    """
    if len(points) < smooth + 4:
        return float("nan")
    kernel = np.ones(smooth) / smooth
    xs = np.convolve(points[:, 0], kernel, mode="valid")
    ys = np.convolve(points[:, 1], kernel, mode="valid")
    pts = np.stack([xs, ys], axis=1)
    # close the curve so wrap-around works for closed contours
    pts = np.vstack([pts, pts[:smooth]])
    tangents = np.diff(pts, axis=0)
    angles = np.arctan2(tangents[:, 1], tangents[:, 0])
    angles = np.unwrap(angles)
    d_angle = np.diff(angles)
    return float(np.mean(d_angle**2))
