"""Synthetic damage mask tests (SPEC.md §11, Faz 1 acceptance).

Checks: deterministic seeds, chip area fractions + contour contact, holes fully
inside the object, cracks thin and inside the object.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from counterpart.config import DamageCfg
from counterpart.data.masks import _boundary_band, crack, hole, sample_damage


@pytest.fixture()
def cfg() -> DamageCfg:
    return DamageCfg()


def ellipse_object(size: int = 256) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 3, size // 2 - 10), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def test_determinism(cfg: DamageCfg) -> None:
    obj = ellipse_object()
    for dtype in ("chip", "hole", "crack"):
        m1, i1 = sample_damage(obj, cfg, np.random.default_rng(42), forced_type=dtype)
        m2, i2 = sample_damage(obj, cfg, np.random.default_rng(42), forced_type=dtype)
        assert np.array_equal(m1, m2), dtype
        assert i1.damage_type == i2.damage_type


def test_chip_area_and_contact(cfg: DamageCfg) -> None:
    obj = ellipse_object()
    band = _boundary_band(obj, 2)
    fracs = []
    for seed in range(30):
        mask, info = sample_damage(obj, cfg, np.random.default_rng(seed), forced_type="chip")
        assert mask.any()
        assert (mask & ~obj).sum() == 0, "chip must stay inside the object"
        assert (mask & band).any(), "chip must touch the object boundary"
        fracs.append(info.area_frac)

    fracs_arr = np.array(fracs)
    in_range = (fracs_arr >= cfg.area_frac_min) & (fracs_arr <= cfg.area_frac_max)
    assert in_range.mean() >= 0.9, f"only {in_range.mean():.0%} of chips hit the target area band"
    assert (fracs_arr <= cfg.area_frac_max * 1.3).all()


def test_hole_inside_and_area(cfg: DamageCfg) -> None:
    obj = ellipse_object()
    for seed in range(30):
        mask, info = hole(obj, np.random.default_rng(seed), cfg)
        assert mask.any()
        assert (mask & ~obj).sum() == 0
        assert not (mask & _boundary_band(obj, 1)).any(), "hole must not touch the boundary"
        assert cfg.area_frac_min * 0.8 <= info["area_frac"] <= cfg.area_frac_max * 1.25


def test_crack_thin(cfg: DamageCfg) -> None:
    obj = ellipse_object()
    obj_area = obj.sum()
    for seed in range(10):
        mask, info = crack(obj, np.random.default_rng(seed), cfg)
        assert mask.any()
        assert (mask & ~obj).sum() == 0
        assert mask.sum() / obj_area < 0.1, "cracks should stay thin"
