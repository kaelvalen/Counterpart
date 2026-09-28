"""Score-term sanity tests (SPEC.md §11, Faz 3 acceptance).

Every term must rank the ground-truth completion above deliberately degraded ones.
The degradations are chosen per term family:

* T1 (seam): ground truth beats white-filled, noise-filled and wrong-region-copied
  completions — all three break the seam statistics it measures.
* T6 (silhouette): ground truth beats fills that break the silhouette (white) or
  replace it with a wrongly-oriented boundary (blocky). Fill styles that merely paste
  *opaque* content (noise, wrong copy) restore the silhouette by covering it — T6
  measures contour continuity only, so ties there are expected and correct.
* T4 (texture): ground truth beats white and noise fills on a textured object.
* T5 (frequency): ground truth beats noise (spurious high frequencies) and an
  over-smoothed fill (the classic diffusion failure mode).
T2/T3 are tested on symmetric synthetic shapes once implemented.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.score.base import build_score_sample
from counterpart.score.boundary import BoundaryTerm
from counterpart.score.contour import ContourTerm
from counterpart.score.frequency import FrequencyTerm
from counterpart.score.rotational import RotationalTerm
from counterpart.score.symmetry import SymmetryTerm
from counterpart.score.texture import TextureTerm
from counterpart.types import Sample


def make_scene(tmp_path: Path, size: int = 256) -> tuple[object, dict[str, np.ndarray]]:
    """Intact object + edge chip; returns (score sample, candidate variants)."""
    intact = np.full((size, size, 3), 248, dtype=np.uint8)
    cv2.ellipse(intact, (size // 2 - 8, size // 2), (60, 78), 0, 0, 360, (90, 70, 150), -1)
    cv2.circle(intact, (size // 2 - 8, size // 2 - 20), 18, (200, 160, 60), -1)

    from counterpart.segment.threshold import object_mask

    cfg = Cfg()
    obj = object_mask(intact, cfg.segment)

    damage = np.zeros((size, size), dtype=bool)
    damage[96:160, 168:210] = True
    damage &= obj
    assert damage.any(), "test damage must intersect the object silhouette"

    damaged = intact.copy()
    damaged[damage] = 248

    rng = np.random.default_rng(0)
    noise = intact.copy()
    noise[damage] = rng.integers(0, 256, size=(int(damage.sum()), 3), dtype=np.uint8)
    wrong = intact.copy()
    wrong[damage] = np.roll(intact, shift=48, axis=1)[damage]

    # blocky: an opaque strip whose new boundary runs vertically instead of following
    # the object's edge — the silhouette continuation has the wrong direction
    blocky = damaged.copy()
    blocky[96:160, 168:171] = (90, 70, 150)

    root = tmp_path
    cio.write_image(root / "damaged.png", damaged)
    cio.write_mask(root / "damage_mask.png", damage)
    cio.write_mask(root / "gen_mask.png", damage)
    cio.write_mask(root / "object_mask.png", obj)

    sample = Sample(sample_id="s0", split="test", root=root, gt_hidden=True)
    ctx = build_score_sample(sample, cfg)
    candidates = {
        "gt": intact,
        "white": damaged,
        "noise": noise,
        "wrong_copy": wrong,
        "blocky": blocky,
    }
    return ctx, candidates


def _scores(term_cls: type, ctx: object, candidates: dict[str, np.ndarray]) -> dict:
    term = term_cls()
    prepared = term.prepare(ctx)
    return {name: term.score(prepared, cand) for name, cand in candidates.items()}


def test_boundary_term_gt_beats_all_degraded(tmp_path: Path) -> None:
    ctx, candidates = make_scene(tmp_path)
    scores = _scores(BoundaryTerm, ctx, candidates)
    assert scores["gt"].applicable
    for name in ("white", "noise", "wrong_copy", "blocky"):
        assert scores["gt"].value > scores[name].value, (
            f"T1: gt ({scores['gt'].value:.4f}) should beat {name} ({scores[name].value:.4f})"
        )


def test_contour_term_gt_beats_silhouette_breaks(tmp_path: Path) -> None:
    ctx, candidates = make_scene(tmp_path)
    scores = _scores(ContourTerm, ctx, candidates)
    assert scores["gt"].applicable
    for name in ("white", "blocky"):
        assert scores["gt"].value > scores[name].value, (
            f"T6: gt ({scores['gt'].value:.4f}) should beat {name} ({scores[name].value:.4f})"
        )


def test_contour_term_abstains_for_interior_hole(tmp_path: Path) -> None:
    """A hole that does not touch the silhouette must make T6 abstain (SPEC.md §5.4)."""
    ctx, candidates = make_scene(tmp_path)
    # rebuild the surface with an interior hole instead of the edge chip
    interior = np.zeros_like(ctx.damage_mask)
    interior[110:140, 110:140] = True
    interior &= ctx.object_mask
    if not interior.any():
        pytest.skip("interior region misses the object")
    ctx.damage_mask = interior
    from counterpart.score.geometry import distance_transforms, rings

    ctx.dist_in, ctx.dist_out = distance_transforms(interior)
    ctx.ring_in, ctx.ring_out, ctx.visible = rings(
        interior, ctx.object_mask, width=ctx.score_cfg.ring_width_px
    )
    term = ContourTerm()
    prepared = term.prepare(ctx)
    result = term.score(prepared, candidates["gt"])
    assert not result.applicable


def test_build_score_sample_blocks_ground_truth(tmp_path: Path) -> None:
    visible = Sample(sample_id="s0", split="test", root=tmp_path, gt_hidden=False)
    with pytest.raises(ValueError):
        build_score_sample(visible, Cfg())


def make_textured_scene(tmp_path: Path, size: int = 384) -> tuple[object, dict[str, np.ndarray]]:
    """Larger textured object with an interior rectangular damage (for T4/T5)."""
    from counterpart.segment.threshold import object_mask

    cfg = Cfg()
    rng = np.random.default_rng(3)
    intact = np.full((size, size, 3), 248, dtype=np.uint8)
    cx, cy = size // 2, size // 2 + 8
    ax, ay = 110, 140
    yy, xx = np.mgrid[0:size, 0:size]
    inside = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0

    body = np.zeros((size, size, 3), dtype=np.float64)
    stripes = 40.0 * np.sin(xx / 3.2) + 20.0 * np.cos(yy / 5.1)
    body[..., 0] = 120 + stripes
    body[..., 1] = 90 + stripes * 0.6
    body[..., 2] = 150 + stripes * 0.3
    body += rng.normal(0.0, 6.0, size=(size, size, 3))
    intact[inside] = np.clip(body[inside], 0, 255).astype(np.uint8)

    obj = object_mask(intact, cfg.segment)
    damage = np.zeros((size, size), dtype=bool)
    damage[150:250, 150:240] = True
    damage &= obj
    assert damage.sum() > 5000

    damaged = intact.copy()
    damaged[damage] = 248
    noise = intact.copy()
    noise[damage] = rng.integers(0, 256, size=(int(damage.sum()), 3), dtype=np.uint8)

    blurred = intact.astype(np.float32)
    blurred = cv2.GaussianBlur(blurred, (0, 0), 6.0)
    smooth = intact.copy()
    smooth[damage] = blurred.astype(np.uint8)[damage]

    root = tmp_path
    cio.write_image(root / "damaged.png", damaged)
    cio.write_mask(root / "damage_mask.png", damage)
    cio.write_mask(root / "gen_mask.png", damage)
    cio.write_mask(root / "object_mask.png", obj)

    sample = Sample(sample_id="s1", split="test", root=root, gt_hidden=True)
    ctx = build_score_sample(sample, cfg)
    candidates = {"gt": intact, "white": damaged, "noise": noise, "smooth": smooth}
    return ctx, candidates


@pytest.mark.parametrize("term_cls", [TextureTerm, FrequencyTerm])
def test_texture_frequency_gt_beats_noise(tmp_path: Path, term_cls: type) -> None:
    ctx, candidates = make_textured_scene(tmp_path)
    scores = _scores(term_cls, ctx, candidates)
    assert scores["gt"].applicable, f"{term_cls.__name__} must be applicable here"
    assert scores["gt"].value > scores["noise"].value, (
        f"{term_cls.__name__}: gt ({scores['gt'].value:.4f}) should beat noise "
        f"({scores['noise'].value:.4f})"
    )


def test_texture_gt_beats_white(tmp_path: Path) -> None:
    ctx, candidates = make_textured_scene(tmp_path)
    scores = _scores(TextureTerm, ctx, candidates)
    assert scores["gt"].value > scores["white"].value


def test_frequency_gt_beats_over_smooth(tmp_path: Path) -> None:
    ctx, candidates = make_textured_scene(tmp_path)
    scores = _scores(FrequencyTerm, ctx, candidates)
    assert scores["gt"].value > scores["smooth"].value, (
        f"T5: gt ({scores['gt'].value:.4f}) should beat an over-smoothed fill "
        f"({scores['smooth'].value:.4f})"
    )


def _scene_skeleton(tmp_path: Path, tile):
    cfg = Cfg()
    root = tmp_path
    cio.write_image(root / "damaged.png", tile["damaged"])
    cio.write_mask(root / "damage_mask.png", tile["damage"])
    cio.write_mask(root / "gen_mask.png", tile["damage"])
    cio.write_mask(root / "object_mask.png", tile["obj"])
    sample = Sample(sample_id="s2", split="test", root=root, gt_hidden=True)
    return build_score_sample(sample, cfg)


def make_mirror_scene(tmp_path: Path, size: int = 320):
    """Ellipse symmetric about the vertical axis, chipped on the right side (T2)."""
    from counterpart.segment.threshold import object_mask

    intact = np.full((size, size, 3), 248, dtype=np.uint8)
    cv2.ellipse(intact, (size // 2, size // 2), (70, 100), 0, 0, 360, (80, 90, 160), -1)
    cv2.circle(intact, (size // 2, size // 2 - 30), 22, (200, 150, 60), -1)

    obj = object_mask(intact, Cfg().segment)
    damage = np.zeros((size, size), dtype=bool)
    damage[110:210, 210:280] = True
    damage &= obj
    damaged = intact.copy()
    damaged[damage] = 248

    flat = damaged.copy()
    flat[damage] = (40, 110, 60)  # mismatching colour
    # dent: filled, but with a bite taken out of the reconstructed silhouette area —
    # the mirror-expected silhouette says "object here", the candidate says no
    dent = damaged.copy()
    dent[damage] = (80, 90, 160)
    cv2.circle(dent, (218, 160), 24, (248, 248, 248), -1)

    ctx = _scene_skeleton(tmp_path, {"damaged": damaged, "damage": damage, "obj": obj})
    return ctx, {"gt": intact, "white": damaged, "flat": flat, "dent": dent}


def make_rotational_scene(tmp_path: Path, size: int = 320):
    """Concentric target with 6 spokes, chipped on the right side (T3)."""
    from counterpart.segment.threshold import object_mask

    intact = np.full((size, size, 3), 248, dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(intact, center, 90, (60, 80, 150), -1)
    cv2.circle(intact, center, 70, (225, 225, 235), -1)
    cv2.circle(intact, center, 50, (60, 80, 150), -1)
    cv2.circle(intact, center, 28, (200, 160, 60), -1)
    for angle_deg in range(0, 360, 60):
        angle = np.deg2rad(angle_deg)
        end = (int(center[0] + 88 * np.cos(angle)), int(center[1] + 88 * np.sin(angle)))
        cv2.line(intact, center, end, (30, 40, 90), 5)

    obj = object_mask(intact, Cfg().segment)
    damage = np.zeros((size, size), dtype=bool)
    damage[120:200, 215:285] = True
    damage &= obj
    damaged = intact.copy()
    damaged[damage] = 248

    flat = damaged.copy()
    flat[damage] = (150, 150, 150)

    ctx = _scene_skeleton(tmp_path, {"damaged": damaged, "damage": damage, "obj": obj})
    return ctx, {"gt": intact, "white": damaged, "flat": flat}


def test_symmetry_term_gt_beats_degraded(tmp_path: Path) -> None:
    ctx, candidates = make_mirror_scene(tmp_path)
    scores = _scores(SymmetryTerm, ctx, candidates)
    assert scores["gt"].applicable, "T2 must find the mirror axis of the ellipse"
    for name in ("white", "flat", "dent"):
        assert scores["gt"].value > scores[name].value, (
            f"T2: gt ({scores['gt'].value:.4f}) should beat {name} ({scores[name].value:.4f})"
        )


def test_rotational_term_gt_beats_degraded(tmp_path: Path) -> None:
    ctx, candidates = make_rotational_scene(tmp_path)
    scores = _scores(RotationalTerm, ctx, candidates)
    assert scores["gt"].applicable, "T3 must detect the rotational symmetry of the target"
    assert scores["gt"].diagnostics["k"] == 6, (
        f"expected 6-fold symmetry, got k={scores['gt'].diagnostics['k']}"
    )
    for name in ("white", "flat"):
        assert scores["gt"].value > scores[name].value, (
            f"T3: gt ({scores['gt'].value:.4f}) should beat {name} ({scores[name].value:.4f})"
        )
