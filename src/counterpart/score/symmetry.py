"""T2 — mirror (bilateral) symmetry consistency (SPEC.md §5.4 T2).

prepare: search for a mirror axis of the object — angles θ ∈ [0,180) with a
coarse-to-fine scan plus small offsets around the silhouette centroid. The shape
term reflects the **full object silhouette** (the damage may hide part of the object,
but the mask still describes where it is); the colour term reflects **visible**
pixels only. If the best axis confidence is below the configured threshold the term
abstains (``applicable=False``).

score: pixels of M whose mirror image lands in V are compared with their mirrored
visible counterparts (Lab ΔE); the candidate's silhouette inside M is compared with
the mirrored silhouette (IoU), including where the mirror expectation is "no object"
(over-fills are penalised too). Value = −c · (w_colour·bad + w_shape·bad).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from counterpart.score.base import ScoreSample, register
from counterpart.segment.threshold import delta_e76, rgb_to_lab
from counterpart.types import TermResult


def _reflect(points: np.ndarray, theta: float, offset: float) -> np.ndarray:
    """Reflect Nx2 points about the line {x : x·n = offset}, n = (cosθ, sinθ)."""
    normal = np.array([np.cos(theta), np.sin(theta)])
    signed = points @ normal - offset
    return points - 2.0 * signed[:, None] * normal[None, :]


def _sample_points(mask: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(points) > max_points:
        index = rng.choice(len(points), size=max_points, replace=False)
        points = points[index]
    return points


def _lookup(mask: np.ndarray, lab: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-pixel lookup; returns (mask membership, Lab values) for points."""
    h, w = mask.shape
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return mask[ys, xs], lab[ys, xs]


@register
class SymmetryTerm:
    name = "T2"

    # ------------------------------------------------------------------ prepare

    def prepare(self, sample: ScoreSample) -> dict[str, Any]:
        cfg = sample.score_cfg
        rng = np.random.default_rng(0)
        lab = sample.damaged_lab

        shape_points = _sample_points(sample.object_mask, max_points=12000, rng=rng)
        colour_points = _sample_points(sample.visible, max_points=4000, rng=rng)
        if len(shape_points) < cfg.t2_min_overlap_px or len(colour_points) < 64:
            return {
                "applicable": False,
                "reason": "visible region too small",
                "cfg": cfg,
                "sample": sample,
            }

        centroid = shape_points.mean(axis=0)
        object_size = float(np.sqrt(sample.object_mask.sum()))
        h, w = lab.shape[:2]
        colour_xs = np.clip(np.round(colour_points[:, 0]).astype(int), 0, w - 1)
        colour_ys = np.clip(np.round(colour_points[:, 1]).astype(int), 0, h - 1)
        colour_lab = lab[colour_ys, colour_xs]

        def axis_score(theta: float, offset: float) -> float:
            # silhouette overlap on the full silhouette
            reflected = _reflect(shape_points, theta, offset)
            in_obj, _ = _lookup(sample.object_mask, lab, reflected)
            inter = int(in_obj.sum())
            union = 2 * len(shape_points) - inter
            iou = inter / union if union > 0 else 0.0

            # colour consistency of mirrored visible points
            reflected_c = _reflect(colour_points, theta, offset)
            in_vis, mirrored_lab = _lookup(sample.visible, lab, reflected_c)
            if int(in_vis.sum()) < 16:
                return iou
            d_e = float(delta_e76(colour_lab[in_vis], mirrored_lab[in_vis]).mean())
            colour = float(np.clip(1.0 - d_e / cfg.t2_color_scale, 0.0, 1.0))
            return 0.5 * iou + 0.5 * colour

        # coarse angular scan through the centroid
        best = (-1.0, 0.0, 0.0)
        for angle_deg in np.arange(0.0, 180.0, cfg.t2_angle_step_deg):
            theta = np.deg2rad(angle_deg)
            normal = np.array([np.cos(theta), np.sin(theta)])
            score = axis_score(theta, float(centroid @ normal))
            if score > best[0]:
                best = (score, float(angle_deg), float(centroid @ normal))

        # refine angle and offset (SPEC: local improvement)
        score_best, angle_best, offset_best = best
        span = cfg.t2_angle_step_deg
        for _ in range(cfg.t2_refine_iters):
            span = span / 3.0
            for angle_deg in np.arange(angle_best - span, angle_best + span + 1e-6, span / 2):
                theta = np.deg2rad(angle_deg % 180.0)
                normal = np.array([np.cos(theta), np.sin(theta)])
                offset_center = float(centroid @ normal)
                for step in range(-cfg.t2_offset_steps, cfg.t2_offset_steps + 1):
                    offset = offset_center + step * cfg.t2_offset_frac * object_size
                    score = axis_score(theta, offset)
                    if score > score_best:
                        score_best = score
                        angle_best = float(angle_deg % 180.0)
                        offset_best = offset

        confidence = float(np.clip(score_best, 0.0, 1.0))
        if confidence < cfg.t2_confidence_threshold:
            return {
                "applicable": False,
                "reason": "no confident mirror axis",
                "cfg": cfg,
                "confidence": confidence,
                "sample": sample,
            }
        return {
            "applicable": True,
            "cfg": cfg,
            "sample": sample,
            "theta_deg": angle_best,
            "offset": offset_best,
            "confidence": confidence,
        }

    # -------------------------------------------------------------------- score

    def score(self, prepared: dict[str, Any], candidate: np.ndarray) -> TermResult:
        if not prepared.get("applicable", False):
            return TermResult(
                value=float("nan"),
                applicable=False,
                diagnostics={
                    "reason": prepared["reason"],
                    "confidence": prepared.get("confidence"),
                },
            )
        cfg = prepared["cfg"]
        sample: ScoreSample = prepared["sample"]
        theta = float(np.deg2rad(prepared["theta_deg"]))
        offset = float(prepared["offset"])
        confidence = float(prepared["confidence"])

        cand_lab = rgb_to_lab(candidate)
        cand_obj = sample.candidate_object_mask(candidate)
        mask = sample.damage_mask
        ys, xs = np.nonzero(mask)
        points = np.stack([xs, ys], axis=1).astype(np.float64)
        if len(points) == 0:
            return TermResult(
                value=float("nan"), applicable=False, diagnostics={"reason": "empty mask"}
            )

        reflected = _reflect(points, theta, offset)
        in_visible, mirrored_lab = _lookup(sample.visible, sample.damaged_lab, reflected)
        h, w = sample.shape
        in_bounds = (
            (reflected[:, 0] >= 0)
            & (reflected[:, 0] <= w - 1)
            & (reflected[:, 1] >= 0)
            & (reflected[:, 1] <= h - 1)
        )

        # colour: only where the mirror lands on visible object pixels
        colour_bad = float("nan")
        d_e_value = float("nan")
        if int(in_visible.sum()) >= 16:
            cand_values = cand_lab[ys[in_visible], xs[in_visible]]
            d_e_value = float(delta_e76(cand_values, mirrored_lab[in_visible]).mean())
            colour_bad = float(np.clip(d_e_value / cfg.t2_color_scale, 0.0, 1.0))

        # silhouette: everywhere the mirror is inside the image — including where the
        # mirrored expectation is "no object" (an over-fill must be penalised too)
        if int(in_bounds.sum()) < 16:
            return TermResult(
                value=float("nan"),
                applicable=False,
                diagnostics={
                    "reason": "mirror of M leaves the image",
                    "n_shape_matched": int(in_bounds.sum()),
                },
            )
        expected_obj, _ = _lookup(sample.object_mask, sample.damaged_lab, reflected)
        expected_obj = expected_obj[in_bounds]
        cand_at = cand_obj[ys[in_bounds], xs[in_bounds]]
        inter = int((cand_at & expected_obj).sum())
        union = int((cand_at | expected_obj).sum())
        shape_bad = 1.0 - (inter / union if union > 0 else 0.0)

        weights = {"colour": (1.0, colour_bad), "shape": (1.0, shape_bad)}
        active = {key: (wt, b) for key, (wt, b) in weights.items() if np.isfinite(b)}
        total_weight = sum(wt for wt, _ in active.values())
        badness = sum(wt * b for wt, b in active.values()) / max(total_weight, 1e-9)
        return TermResult(
            value=-confidence * badness,
            applicable=True,
            diagnostics={
                "confidence": confidence,
                "mean_delta_e": d_e_value,
                "colour_bad": colour_bad,
                "shape_bad": shape_bad,
                "n_colour_matched": int(in_visible.sum()),
                "n_shape_matched": int(in_bounds.sum()),
                "theta_deg": prepared["theta_deg"],
            },
        )
