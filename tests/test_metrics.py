"""Metric sanity tests (SPEC.md §11): perfect candidate must beat damaged ones."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from counterpart.config import EvalCfg, SegmentCfg
from counterpart.eval.metrics import (
    boundary_delta_e,
    crop_box_for_mask,
    masked_psnr,
    reconstruction_metrics,
)
from counterpart.segment.threshold import object_mask


def _scene(size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """White background, dark ellipse object, rectangular damage region inside it."""
    gt = np.full((size, size, 3), 250, dtype=np.uint8)
    cv2.ellipse(gt, (size // 2, size // 2), (size // 4, size // 3), 0, 0, 360, (70, 80, 130), -1)
    cv2.circle(gt, (size // 2, size // 3), size // 16, (200, 150, 60), -1)
    obj = object_mask(gt, SegmentCfg())
    damage = np.zeros((size, size), dtype=bool)
    damage[size // 2 : size // 2 + 30, size // 2 : size // 2 + 30] = True
    damage &= obj
    assert damage.any()
    return gt, damage, obj


def test_perfect_candidate_scores_best() -> None:
    gt, damage, obj = _scene()
    rng = np.random.default_rng(0)

    perfect = gt.copy()
    white_filled = gt.copy()
    white_filled[damage] = 250
    noisy = gt.copy()
    noisy[damage] = rng.integers(0, 256, size=(int(damage.sum()), 3), dtype=np.uint8)

    cfg = EvalCfg()
    seg = SegmentCfg()
    m_perfect = reconstruction_metrics(gt, perfect, damage, obj, cfg, seg)
    m_white = reconstruction_metrics(gt, white_filled, damage, obj, cfg, seg)
    m_noisy = reconstruction_metrics(gt, noisy, damage, obj, cfg, seg)

    assert m_perfect["lpips"] < 1e-4
    assert m_perfect["lpips"] < m_white["lpips"]
    assert m_perfect["lpips"] < m_noisy["lpips"]
    assert m_perfect["psnr"] == float("inf")
    assert m_white["psnr"] < m_perfect["psnr"]
    assert m_perfect["ssim"] > m_white["ssim"]
    assert m_perfect["delta_e_boundary"] < 1e-3
    assert m_white["delta_e_boundary"] > m_perfect["delta_e_boundary"]
    assert m_perfect["silhouette_iou"] >= 0.999


def test_masked_psnr_inf_for_identical() -> None:
    gt, damage, _ = _scene()
    assert masked_psnr(gt, gt, damage) == float("inf")


def test_crop_box_respects_bounds_and_min_side() -> None:
    mask = np.zeros((100, 140), dtype=bool)
    mask[10:20, 130:140] = True
    y0, y1, x0, x1 = crop_box_for_mask(mask, margin=16, min_side=64)
    assert 0 <= y0 < y1 <= 100
    assert 0 <= x0 < x1 <= 140
    assert (y1 - y0) >= 36 and (x1 - x0) >= 26


def test_boundary_delta_e_zero_for_identical() -> None:
    gt, damage, _ = _scene()
    assert boundary_delta_e(gt, gt, damage, ring_px=6) == pytest.approx(0.0, abs=1e-6)
