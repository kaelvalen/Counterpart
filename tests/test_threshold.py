"""Segmentation tests on synthetic white-background scenes."""

from __future__ import annotations

import cv2
import numpy as np

from counterpart.config import SegmentCfg
from counterpart.segment.threshold import (
    background_whiteness,
    check_object_mask,
    crop_and_resize,
    object_mask,
    square_crop_box,
)


def synthetic_scene(size: int = 600) -> np.ndarray:
    """White background with a dark ellipse object in the middle."""
    img = np.full((size, size, 3), 250, dtype=np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 5, size // 4), 0, 0, 360, (60, 70, 120), -1)
    return img


def test_object_mask_finds_object() -> None:
    cfg = SegmentCfg()
    img = synthetic_scene()
    mask = object_mask(img, cfg)
    ok, frac = check_object_mask(mask, cfg)
    assert ok, frac
    assert 0.10 < frac < 0.30, frac
    # mask centre inside object, corner outside
    assert mask[300, 300]
    assert not mask[5, 5]


def test_background_whiteness() -> None:
    cfg = SegmentCfg()
    assert background_whiteness(synthetic_scene(), cfg) == 1.0
    noisy = synthetic_scene()
    rng = np.random.default_rng(0)
    noisy[:20] = rng.integers(0, 256, size=noisy[:20].shape, dtype=np.uint8)
    assert background_whiteness(noisy, cfg) < 0.95


def test_crop_box_and_resize() -> None:
    cfg = SegmentCfg(size=256)
    img = synthetic_scene(size=600)
    mask = object_mask(img, cfg)
    x0, y0, side = square_crop_box(mask, margin=0.15)
    assert side <= 600 and x0 >= 0 and y0 >= 0
    img_r, mask_r, box = crop_and_resize(img, mask, cfg)
    assert img_r.shape == (256, 256, 3)
    assert mask_r.shape == (256, 256)
    assert mask_r.any() and not mask_r.all()
    assert box == (x0, y0, side)
