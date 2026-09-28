"""Seam blending tests: exact outside, exact inside, ramp in the feather band."""

from __future__ import annotations

import numpy as np

from counterpart.generate.sd_inpaint import blend_seam, dilate_mask


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cand = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True
    return img, cand, mask


def test_feather_zero_is_hard_replace() -> None:
    img, cand, mask = _scene()
    out = blend_seam(img, cand, mask, 0)
    assert np.array_equal(out[mask], cand[mask])
    assert np.array_equal(out[~mask], img[~mask])


def test_outside_dilate_is_exact_original() -> None:
    img, cand, mask = _scene()
    feather = 3
    out = blend_seam(img, cand, mask, feather)
    outside = ~dilate_mask(mask, feather)
    assert np.array_equal(out[outside], img[outside]), "outside band must be bit-exact original"


def test_inside_erode_is_exact_candidate() -> None:
    import cv2

    img, cand, mask = _scene()
    out = blend_seam(img, cand, mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    core = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    assert np.array_equal(out[core], cand[core])


def test_dilate_mask_grows_region() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[15, 15] = True
    grown = dilate_mask(mask, 2)
    assert grown.sum() > mask.sum()
    assert grown[15, 17]  # 2 px to the right: inside the 2px dilation
    assert not grown[15, 18]  # 3 px away: outside
