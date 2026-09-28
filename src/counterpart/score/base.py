"""Score-term protocol and the ground-truth-free sample view (SPEC.md §5.4, §13).

Score terms may only see the damaged input and the masks — never ``original.png``.
The runner builds a :class:`ScoreSample` from a ``gt_hidden`` sample view and passes
it to each term's ``prepare``; ``score`` then evaluates candidates against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from counterpart import io as cio
from counterpart.config import Cfg, ScoreCfg
from counterpart.score.geometry import (
    boundary_points,
    distance_transforms,
    rings,
    signed_distance,
)
from counterpart.segment.threshold import rgb_to_lab
from counterpart.types import Sample, TermResult


@dataclass(slots=True)
class ScoreSample:
    """Everything a score term may look at; ground truth is absent by construction."""

    sample_id: str
    damaged: np.ndarray  # RGB uint8 HxWx3
    damage_mask: np.ndarray  # M
    gen_mask: np.ndarray  # dilated M (what the generator repainted)
    object_mask: np.ndarray  # object silhouette
    damaged_lab: np.ndarray  # float32 Lab of damaged
    dist_in: np.ndarray  # distance to boundary, inside M
    dist_out: np.ndarray  # distance to boundary, outside M
    ring_in: np.ndarray  # R_in
    ring_out: np.ndarray  # R_out
    visible: np.ndarray  # V = object \ dilate(M, w)
    score_cfg: ScoreCfg
    seg_cfg: Any
    meta: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return self.damaged.shape[:2]

    def candidate_object_mask(self, candidate: np.ndarray) -> np.ndarray:
        """Silhouette of a candidate via background thresholding (synthetic mode)."""
        from counterpart.segment.threshold import object_mask

        return object_mask(candidate, self.seg_cfg)


def build_score_sample(sample: Sample, cfg: Cfg) -> ScoreSample:
    """Load the damaged input + masks (never GT) into a :class:`ScoreSample`."""
    if not sample.gt_hidden:
        raise ValueError("build_score_sample requires a gt_hidden sample view (SPEC.md §13)")
    damaged = cio.read_image(sample.damaged_path)
    damage_mask = cio.read_mask(sample.damage_mask_path)
    gen_mask = cio.read_mask(sample.gen_mask_path)
    object_mask = cio.read_mask(sample.object_mask_path)

    dist_in, dist_out = distance_transforms(damage_mask)
    ring_in, ring_out, visible = rings(damage_mask, object_mask, width=cfg.score.ring_width_px)
    return ScoreSample(
        sample_id=sample.sample_id,
        damaged=damaged,
        damage_mask=damage_mask,
        gen_mask=gen_mask,
        object_mask=object_mask,
        damaged_lab=rgb_to_lab(damaged),
        dist_in=dist_in,
        dist_out=dist_out,
        ring_in=ring_in,
        ring_out=ring_out,
        visible=visible,
        score_cfg=cfg.score,
        seg_cfg=cfg.segment,
        meta=sample.meta,
    )


class ScoreTerm(Protocol):
    """SPEC.md §5.4 protocol: ``value`` is high = good, ``applicable`` gates weighting."""

    name: str

    def prepare(self, sample: ScoreSample) -> Any: ...

    def score(self, prepared: Any, candidate: np.ndarray) -> TermResult: ...


# A small registry so the CLI can select terms by name.
TERMS: dict[str, type] = {}


def register(cls: type) -> type:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls} must define a class-level 'name'")
    TERMS[name] = cls  # type: ignore[assignment]
    return cls


def boundary_normals(sample: ScoreSample, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Contour sample points of M plus outward unit normals (SPEC.md §5.4 T1)."""
    points = boundary_points(sample.damage_mask, stride)
    sdf = signed_distance(sample.dist_in, sample.dist_out)
    return points, normals_at(sdf, points)


def normals_at(sdf: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Outward normals at integer pixel positions from the signed distance gradient."""
    import cv2

    gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3)
    h, w = sdf.shape
    normals = []
    for x, y in points:
        ix = int(np.clip(x, 1, w - 2))
        iy = int(np.clip(y, 1, h - 2))
        vx, vy = float(gx[iy, ix]), float(gy[iy, ix])
        n = np.hypot(vx, vy)
        if n < 1e-6:
            # fall back to the direction away from the mask centroid
            ys, xs = np.nonzero(sdf < 0)
            vx, vy = float(ix - xs.mean()), float(iy - ys.mean())
            n = max(float(np.hypot(vx, vy)), 1e-6)
        normals.append((vx / n, vy / n))
    return np.asarray(normals, dtype=np.float64)


def sample_band(
    image: np.ndarray, points: np.ndarray, normals: np.ndarray, depths: np.ndarray
) -> np.ndarray:
    """Sample ``image`` at ``point + normal * depth`` for every depth; returns NxKxC.

    Out-of-bounds samples are clamped to the image edge (the defects are interior, so
    this only matters for degenerate masks).
    """
    h, w = image.shape[:2]
    coords = points[:, None, :] + normals[:, None, :] * depths[None, :, None]
    xs = np.clip(np.round(coords[..., 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(coords[..., 1]).astype(int), 0, h - 1)
    if image.ndim == 2:
        return image[ys, xs]
    return image[ys, xs]
