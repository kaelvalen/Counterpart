"""T3 — rotational symmetry consistency (SPEC.md §5.4 T3).

prepare: the rotation centre starts from silhouette moments and is refined with a
circular Hough transform (when one is found), then by a small offset grid search.
A continuous polar resampling of the damaged image (bilinear) provides the reference
colour map — no histogram binning, so radial colour structure is preserved exactly.
Angular fold consistency over k ∈ [k_min, k_max] selects the fundamental k (the
largest k within tolerance of the best); radially uniform objects are flagged as
continuously rotationally symmetric.

score: candidate pixels inside M are compared (Lab ΔE) with the mean of their
same-radius reference counterparts rotated by 2π/k. Value = −c_r · mean ΔE; abstains
below the confidence threshold.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from counterpart.score.base import ScoreSample, register
from counterpart.segment.threshold import delta_e76, rgb_to_lab
from counterpart.types import TermResult

K_TOLERANCE = 0.05  # consistency gap within which a larger k is preferred (fundamental period)


def _polar_grid(
    center: np.ndarray, r_max: float, n_radial: int, n_angular: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radii = (np.arange(n_radial) + 0.5) * (r_max / n_radial)
    angles = np.arange(n_angular, dtype=np.float64) * (2.0 * np.pi / n_angular)
    xx = center[0] + radii[:, None] * np.cos(angles)[None, :]
    yy = center[1] + radii[:, None] * np.sin(angles)[None, :]
    return xx.astype(np.float32), yy.astype(np.float32), angles


def _polar_resample(
    image: np.ndarray,
    center: np.ndarray,
    r_max: float,
    n_radial: int,
    n_angular: int,
    *,
    nearest: bool = False,
) -> np.ndarray:
    xx, yy, _ = _polar_grid(center, r_max, n_radial, n_angular)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(
        image.astype(np.float32),
        xx,
        yy,
        interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _angular_shift(polar: np.ndarray, shift_bins_float: float) -> np.ndarray:
    """Shift a polar map along the angle axis (wrapping); exact roll for integer shifts."""
    n_angular = polar.shape[1]
    shift = shift_bins_float % n_angular
    if abs(shift - round(shift)) < 1e-9:
        return np.roll(polar, -int(round(shift)), axis=1)
    angles = np.arange(n_angular, dtype=np.float64) * (2.0 * np.pi / n_angular)
    targets = (angles + shift * (2.0 * np.pi / n_angular)) % (2.0 * np.pi)
    out = np.empty_like(polar)
    if polar.ndim == 2:
        for r in range(polar.shape[0]):
            out[r] = np.interp(targets, angles, polar[r], period=2.0 * np.pi)
    else:
        for r in range(polar.shape[0]):
            for c in range(polar.shape[2]):
                out[r, :, c] = np.interp(targets, angles, polar[r, :, c], period=2.0 * np.pi)
    return out


def _fold_consistency(
    polar_lab: np.ndarray,
    polar_visible: np.ndarray,
    k: int,
    color_scale: float,
) -> float:
    """Mean ΔE between each visible polar cell and its k-fold rotated counterparts."""
    n_angular = polar_lab.shape[1]
    acc = np.zeros_like(polar_lab)
    valid = np.zeros(polar_visible.shape, dtype=np.float64)
    for j in range(1, k):
        shift = j * n_angular / k
        shifted_visible = _angular_shift(polar_visible.astype(np.float64), shift)
        # only visible counterparts contribute — otherwise damaged (white) cells
        # would pollute the mean while not counting in the denominator
        acc += _angular_shift(polar_lab, shift) * shifted_visible[:, :, None]
        valid += shifted_visible
    ok = (valid > 0) & (polar_visible > 0)
    if int(ok.sum()) < 32:
        return float("nan")
    predicted = acc[ok] / valid[ok, None]
    d_e = delta_e76(polar_lab[ok], predicted)
    return float(1.0 - np.clip(np.mean(d_e) / color_scale, 0.0, 1.0))


@register
class RotationalTerm:
    name = "T3"

    # ------------------------------------------------------------------ prepare

    def prepare(self, sample: ScoreSample) -> dict[str, Any]:
        cfg = sample.score_cfg
        n_radial, n_angular = cfg.t3_radial_bins, cfg.t3_angular_bins
        visible = sample.visible
        lab = sample.damaged_lab

        ys, xs = np.nonzero(visible)
        if len(xs) < cfg.t2_min_overlap_px:
            return {
                "applicable": False,
                "reason": "visible region too small",
                "cfg": cfg,
                "sample": sample,
            }

        # --- centre: silhouette moments, refined by a circular Hough transform
        moments = cv2.moments(sample.object_mask.astype(np.uint8))
        if moments["m00"] > 0:
            center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        else:
            center = np.array([xs.mean(), ys.mean()])

        def radius_extent(centre: np.ndarray) -> float:
            return (
                float(np.hypot(xs - centre[0], ys - centre[1]).max() * 1.02)
                if len(xs)
                else float(min_dim)
            )

        def consistency_at(centre: np.ndarray, k: int, r_max: float) -> float:
            polar_lab = _polar_resample(lab, centre, r_max, n_radial, n_angular)
            polar_vis = _polar_resample(
                visible.astype(np.float32), centre, r_max, n_radial, n_angular, nearest=True
            )
            return _fold_consistency(polar_lab, polar_vis > 0.5, k, cfg.t3_color_scale)

        gray = cv2.cvtColor(sample.damaged, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        min_dim = min(sample.shape)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim // 2,
            param1=120,
            param2=40,
            minRadius=int(cfg.t3_hough_min_radius_frac * min_dim),
            maxRadius=int(cfg.t3_hough_max_radius_frac * min_dim),
        )
        hough_used = False
        if circles is not None and len(circles[0]) > 0:
            candidate_center = circles[0][0][:2].astype(np.float64)
            if np.linalg.norm(candidate_center - center) < 0.15 * min_dim:
                # verify: keep the Hough centre only if it explains the pattern at
                # least as well as the moments centre (Hough can lock onto eccentric
                # circles on ringed objects)
                probe_k = cfg.t3_k_min
                r_probe = radius_extent(center)
                base_score = consistency_at(center, probe_k, r_probe)
                hough_score = consistency_at(candidate_center, probe_k, r_probe)
                if np.isfinite(hough_score) and (
                    not np.isfinite(base_score) or hough_score >= base_score
                ):
                    center = candidate_center
                    hough_used = True

        r_max = radius_extent(center)

        # --- k sweep at the initial centre
        scores: dict[int, float] = {}
        for k in range(cfg.t3_k_min, cfg.t3_k_max + 1):
            scores[k] = consistency_at(center, k, r_max)
        best_score = max((v for v in scores.values() if np.isfinite(v)), default=float("nan"))

        # --- centre refinement for a mid k (the pattern is present for all true k)
        refine_k = int(np.nanargmax([scores[k] for k in sorted(scores)])) + cfg.t3_k_min
        for dy in (-6.0, -3.0, 0.0, 3.0, 6.0):
            for dx in (-6.0, -3.0, 0.0, 3.0, 6.0):
                if dx == 0.0 and dy == 0.0:
                    continue
                shifted = center + np.array([dx, dy])
                value = consistency_at(shifted, refine_k, r_max)
                if np.isfinite(value) and value > best_score + 1e-6:
                    best_score = value
                    center = shifted

        # re-evaluate all k at the (possibly refined) centre
        for k in range(cfg.t3_k_min, cfg.t3_k_max + 1):
            value = consistency_at(center, k, r_max)
            scores[k] = value

        finite = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not finite:
            return {
                "applicable": False,
                "reason": "no valid rotational references",
                "cfg": cfg,
                "confidence": 0.0,
                "sample": sample,
            }

        best_score = max(finite.values())
        # fundamental period: largest k whose consistency is within tolerance of the best
        best_k = max(k for k, v in finite.items() if v >= best_score - K_TOLERANCE)
        confidence = float(np.clip(best_score, 0.0, 1.0))
        continuous = bool(
            (max(finite.values()) - min(finite.values())) < cfg.t3_continuous_tol
            and confidence >= cfg.t3_confidence_threshold
        )
        if continuous:
            best_k = cfg.t3_k_min

        if confidence < cfg.t3_confidence_threshold:
            return {
                "applicable": False,
                "reason": "no confident rotational symmetry",
                "cfg": cfg,
                "confidence": confidence,
                "sample": sample,
            }

        polar_lab = _polar_resample(lab, center, r_max, n_radial, n_angular)
        polar_vis = _polar_resample(
            visible.astype(np.float32), center, r_max, n_radial, n_angular, nearest=True
        )
        return {
            "applicable": True,
            "cfg": cfg,
            "sample": sample,
            "center": center,
            "r_max": r_max,
            "k": int(best_k),
            "confidence": confidence,
            "continuous": continuous,
            "hough_used": hough_used,
            "reference_polar": polar_lab,
            "reference_visible": polar_vis > 0.5,
            "k_scores": {int(k): round(float(v), 4) for k, v in scores.items() if np.isfinite(v)},
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
        k = int(prepared["k"])
        n_radial, n_angular = cfg.t3_radial_bins, cfg.t3_angular_bins

        mask = sample.damage_mask & sample.object_mask
        if int(mask.sum()) < 32:
            return TermResult(
                value=float("nan"), applicable=False, diagnostics={"reason": "mask too small"}
            )

        reference = prepared["reference_polar"]  # (R, A, 3) Lab
        ref_visible = prepared["reference_visible"]
        cand_polar = _polar_resample(
            rgb_to_lab(candidate), prepared["center"], prepared["r_max"], n_radial, n_angular
        )
        mask_polar = _polar_resample(
            mask.astype(np.float32),
            prepared["center"],
            prepared["r_max"],
            n_radial,
            n_angular,
            nearest=True,
        )

        acc = np.zeros_like(reference)
        valid = np.zeros(ref_visible.shape, dtype=np.float64)
        for j in range(1, k):
            shift = j * n_angular / k
            shifted_visible = _angular_shift(ref_visible.astype(np.float64), shift)
            acc += _angular_shift(reference, shift) * shifted_visible[:, :, None]
            valid += shifted_visible

        ok = (mask_polar > 0.5) & (valid > 0)
        if int(ok.sum()) < 32:
            return TermResult(
                value=float("nan"),
                applicable=False,
                diagnostics={"reason": "no rotational reference for damaged area"},
            )
        predicted = acc[ok] / valid[ok, None]
        d_e = float(delta_e76(cand_polar[ok], predicted).mean())
        badness = float(np.clip(d_e / cfg.t3_color_scale, 0.0, 1.0))
        return TermResult(
            value=-float(prepared["confidence"]) * badness,
            applicable=True,
            diagnostics={
                "k": k,
                "confidence": prepared["confidence"],
                "continuous": prepared["continuous"],
                "mean_delta_e": d_e,
                "n_cells": int(ok.sum()),
            },
        )
