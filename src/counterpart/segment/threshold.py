"""Object silhouette for white-background product photos (SPEC.md §5.1, synthetic mode).

Background is estimated as the median CIELAB colour of the image border. Pixels whose
ΔE76 to that background exceeds a threshold form the object; morphology + connected
components + hole filling clean the mask. No SAM needed for ABO-style images.

Images: RGB uint8 HxWx3. Masks: bool HxW.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.color import rgb2lab

from counterpart.config import SegmentCfg


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 CIELAB (skimage; central Lab conversion point, SPEC.md §13)."""
    return rgb2lab(image.astype(np.float32) / 255.0).astype(np.float32)


def _border_pixels(lab: np.ndarray, border: int) -> np.ndarray:
    return np.concatenate(
        [
            lab[:border].reshape(-1, 3),
            lab[-border:].reshape(-1, 3),
            lab[:, :border].reshape(-1, 3),
            lab[:, -border:].reshape(-1, 3),
        ]
    )


def estimate_background_lab(lab: np.ndarray, border: int = 8) -> np.ndarray:
    """Median Lab colour of the image border."""
    return np.median(_border_pixels(lab, border), axis=0)


def delta_e76(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIEDE76 colour difference between two Lab colours (broadcasting supported)."""
    return np.sqrt(((lab_a - lab_b) ** 2).sum(axis=-1))


def background_whiteness(image: np.ndarray, cfg: SegmentCfg) -> float:
    """Fraction of border pixels whose ΔE to the median background is small.

    SPEC.md §6.1 filter: a product image counts as "flat background" only if this is
    high (config ``whiteness_frac``, default 95%).
    """
    lab = rgb_to_lab(image)
    bg = estimate_background_lab(lab, cfg.border_px)
    return float((delta_e76(_border_pixels(lab, cfg.border_px), bg) < cfg.whiteness_delta_e).mean())


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.astype(np.uint8).copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    return (mask.astype(np.uint8) | (1 - flood)).astype(bool)


def object_mask(image: np.ndarray, cfg: SegmentCfg) -> np.ndarray:
    """Threshold + close + largest CC + fill holes; returns a bool HxW mask."""
    lab = rgb_to_lab(image)
    bg = estimate_background_lab(lab, cfg.border_px)
    rough = delta_e76(lab, bg) > cfg.background_delta_e

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
    closed = cv2.morphologyEx(rough.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = fill_holes(largest_component(closed.astype(bool)))
    return mask


def check_object_mask(mask: np.ndarray, cfg: SegmentCfg) -> tuple[bool, float]:
    """Quality gate (SPEC.md §5.1): object area fraction within [min, max]."""
    frac = float(mask.mean())
    ok = cfg.min_area_frac <= frac <= cfg.max_area_frac
    return ok, frac


def square_crop_box(mask: np.ndarray, margin: float = 0.15) -> tuple[int, int, int]:
    """Square box ``(x0, y0, side)`` around the object bbox with a relative margin.

    Final crop policy (SPEC.md §5.1): object bbox + margin, made square, clipped to
    the image bounds.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty object mask")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = mask.shape
    side = int(round(max(x1 - x0 + 1, y1 - y0 + 1) * (1.0 + 2 * margin)))
    side = min(side, h, w)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bx = int(np.clip(cx - side // 2, 0, w - side))
    by = int(np.clip(cy - side // 2, 0, h - side))
    return bx, by, side


def crop_and_resize(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: SegmentCfg,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Crop to the object square box and resize to ``cfg.size``.

    Returns (image_rgb, object_mask, (x0, y0, side)) in original pixel coordinates.
    """
    x0, y0, side = square_crop_box(mask, cfg.crop_margin)
    crop = image[y0 : y0 + side, x0 : x0 + side]
    crop_mask = mask[y0 : y0 + side, x0 : x0 + side]
    interp = cv2.INTER_AREA if side > cfg.size else cv2.INTER_LINEAR
    image_r = cv2.resize(crop, (cfg.size, cfg.size), interpolation=interp)
    mask_r = cv2.resize(
        crop_mask.astype(np.uint8), (cfg.size, cfg.size), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return image_r, mask_r, (x0, y0, side)
