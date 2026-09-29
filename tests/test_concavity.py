"""Concavity heuristic tests for the missing-piece demo path (localize/refine.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from counterpart.config import SegmentCfg
from counterpart.localize.refine import (
    concavity_mask,
    missing_piece_mask,
    should_be_object_mask,
    symmetry_gap_mask,
)


def make_vase_with_bite(size: int = 384) -> tuple[np.ndarray, np.ndarray]:
    """Light ceramic vase on a white background with a chunk missing from the left."""
    image = np.full((size, size, 3), 250, dtype=np.uint8)
    vase = np.zeros((size, size), dtype=np.uint8)  # the vase's true extent (pre-cut)
    cv2.ellipse(vase, (size // 2 + 5, size - 120), (85, 105), 0, 0, 360, 1, -1)
    cv2.rectangle(vase, (size // 2 - 30, 70), (size // 2 + 40, 200), 1, -1)
    cv2.ellipse(vase, (size // 2 + 5, 74), (60, 16), 0, 0, 360, 1, -1)
    vase = vase.astype(bool)

    image[vase] = (198, 200, 205)
    left_half = np.arange(size)[None, :] < size // 2
    image[vase & left_half] = (212, 214, 218)
    gradient = np.linspace(-14, 14, size)[None, :, None]
    image = np.clip(image.astype(np.float32) + gradient, 0, 255).astype(np.uint8)

    # the missing piece: cut a blob out of the left flank, back to background
    bite_ellipse = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(bite_ellipse, (size // 2 - 55, size - 190), (42, 58), 20, 0, 360, 1, -1)
    bite = bite_ellipse.astype(bool) & vase
    image[bite] = 250
    return image, bite


def test_concavity_finds_the_missing_piece() -> None:
    image, bite = make_vase_with_bite()
    mask = concavity_mask(image, SegmentCfg())
    assert mask.any()
    # the hull test only recovers the part of a wide cut the new hull does not bridge
    recall = float((mask & bite).sum()) / max(int(bite.sum()), 1)
    assert recall > 0.5, f"the bite must be partly covered (recall={recall:.2f})"


def test_symmetry_gap_finds_the_missing_piece() -> None:
    image, bite = make_vase_with_bite()
    mask = symmetry_gap_mask(image, SegmentCfg())
    recall = float((mask & bite).sum()) / max(int(bite.sum()), 1)
    assert recall > 0.8, f"mirror gap should cover the bite well (recall={recall:.2f})"


def test_missing_piece_mask_union_is_best() -> None:
    image, bite = make_vase_with_bite()
    mask = missing_piece_mask(image, SegmentCfg())
    recall = float((mask & bite).sum()) / max(int(bite.sum()), 1)
    assert recall > 0.9, recall
    assert int(mask.sum()) < 4 * int(bite.sum()), "hint must stay local"


def test_should_be_object_mask_covers_the_bite() -> None:
    image, bite = make_vase_with_bite()
    expected = should_be_object_mask(image, SegmentCfg())
    coverage = float((expected & bite).sum()) / max(int(bite.sum()), 1)
    assert coverage > 0.9, "the expected silhouette must cover the missing region"


def test_non_convex_object_is_rejected() -> None:
    """A chair-like shape has a huge hull gap — the heuristic must refuse."""
    image = np.full((384, 384, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (100, 80), (280, 130), (200, 200, 205), -1)  # seat
    cv2.rectangle(image, (110, 130), (140, 320), (200, 200, 205), -1)  # leg
    cv2.rectangle(image, (240, 130), (270, 320), (200, 200, 205), -1)  # leg
    with pytest.raises(ValueError):
        concavity_mask(image, SegmentCfg())
    with pytest.raises(ValueError):
        symmetry_gap_mask(image, SegmentCfg())
