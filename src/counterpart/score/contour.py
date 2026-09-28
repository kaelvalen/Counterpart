"""T6 — contour (silhouette) continuity across the damage seam (SPEC.md §5.4 T6).

The visible object silhouette is cut by ∂M at "entry/exit" points. A good completion
continues the silhouette from those points: same local direction (tangent) and a
smooth curvature profile inside M. Fills that stop short of the silhouette, or that
create a new, mismatched boundary (jagged or wrongly oriented), are penalised.

Applicable only when the damage actually interrupts the silhouette (chips/notches);
for interior holes the outer contour is untouched and the term abstains.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from counterpart.score.base import ScoreSample, register
from counterpart.score.geometry import (
    angle_between,
    boolean_runs,
    curvature_energy,
    fit_line_direction,
    ordered_contour_points,
    ordered_contours,
)
from counterpart.segment.threshold import object_mask
from counterpart.types import TermResult


def _points_within(points: np.ndarray, center: tuple[float, float], radius: float) -> np.ndarray:
    if len(points) == 0:
        return points
    d = np.hypot(points[:, 0] - center[0], points[:, 1] - center[1])
    return points[d <= radius]


def _entry_exit_contacts(
    contour: np.ndarray, dist_out: np.ndarray, max_dist: float
) -> list[tuple[float, float]]:
    """First/last contour point of every contiguous run adjacent to ∂M."""
    h, w = dist_out.shape
    adjacent = np.array(
        [
            dist_out[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))] <= max_dist
            for x, y in contour
        ],
        dtype=bool,
    )
    n = len(contour)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if adjacent[i]:
            start = i
            while i < n and adjacent[i]:
                i += 1
            runs.append((start, i - 1))
        else:
            i += 1
    # wraparound: the contour is closed, merge a run spanning index 0
    if len(runs) > 1 and adjacent[0] and adjacent[-1]:
        last, first = runs[-1], runs[0]
        runs = runs[1:-1] + [(last[0], first[1] + n)]

    contacts: list[tuple[float, float]] = []
    for start, end in runs:
        for index in {start, end}:
            x, y = contour[index % n]
            point = (float(x), float(y))
            if all(np.hypot(point[0] - cx, point[1] - cy) > 2 for cx, cy in contacts):
                contacts.append(point)
    return contacts


@register
class ContourTerm:
    name = "T6"

    # ------------------------------------------------------------------ prepare

    def prepare(self, sample: ScoreSample) -> dict[str, Any]:
        cfg = sample.score_cfg
        visible_object = sample.object_mask & ~sample.damage_mask
        try:
            vis_contour = ordered_contour_points(visible_object.astype(bool))
        except ValueError:
            return {"applicable": False, "reason": "no visible object contour", "cfg": cfg}

        contacts = _entry_exit_contacts(vis_contour, sample.dist_out, cfg.t6_contact_px)
        if not contacts:
            return {"applicable": False, "reason": "silhouette not interrupted", "cfg": cfg}

        # outside reference direction: silhouette points strictly away from the cut
        h, w = sample.shape
        far = np.array(
            [
                sample.dist_out[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))]
                > cfg.t6_contact_px
                for x, y in vis_contour
            ],
            dtype=bool,
        )
        far_contour = vis_contour[far] if far.any() else vis_contour

        out_dirs: list[np.ndarray | None] = []
        for contact in contacts:
            pts = _points_within(far_contour, contact, cfg.t6_fit_radius_px)
            out_dirs.append(fit_line_direction(pts) if len(pts) >= 4 else None)

        return {
            "cfg": cfg,
            "seg_cfg": sample.seg_cfg,
            "damage_mask": sample.damage_mask,
            "dist_in": sample.dist_in,
            "contacts": contacts,
            "out_dirs": out_dirs,
            "applicable": True,
        }

    # -------------------------------------------------------------------- score

    def score(self, prepared: dict[str, Any], candidate: np.ndarray) -> TermResult:
        if not prepared.get("applicable", False):
            return TermResult(
                value=float("nan"), applicable=False, diagnostics={"reason": prepared["reason"]}
            )

        cfg = prepared["cfg"]
        damage_mask = prepared["damage_mask"]
        dist_in = prepared["dist_in"]
        contacts: list[tuple[float, float]] = prepared["contacts"]
        out_dirs = prepared["out_dirs"]

        cand_obj = object_mask(candidate, prepared["seg_cfg"])

        # generated silhouette = candidate object-boundary pixels lying strictly inside M
        # (boundary pixels on ∂M itself are cut artifacts of the damage mask)
        h, w = damage_mask.shape
        runs: list[np.ndarray] = []
        for contour in ordered_contours(cand_obj):
            flags = np.array(
                [
                    dist_in[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))] >= 1
                    for x, y in contour
                ],
                dtype=bool,
            )
            if not flags.any():
                continue
            n = len(contour)
            for start, end in boolean_runs(flags):
                index = np.arange(start, end + 1) % n
                run = contour[index]
                if len(run) >= 3:
                    runs.append(run)
        generated = np.concatenate(runs) if runs else np.empty((0, 2))

        penalties: list[float] = []
        fits = 0
        for contact, out_dir in zip(contacts, out_dirs, strict=True):
            pts = _points_within(generated, contact, cfg.t6_fit_radius_px)
            if len(pts) < 4 or out_dir is None:
                penalties.append(1.0)
                continue
            deviation = angle_between(out_dir, fit_line_direction(pts))
            penalties.append(float(np.clip(deviation / (np.pi / 2), 0.0, 1.0)))
            fits += 1
        tangent_bad = float(np.mean(penalties)) if penalties else float("nan")

        # curvature smoothness of the generated silhouette segments
        curvature_bad = float("nan")
        energies = [curvature_energy(run) for run in runs if len(run) >= cfg.t6_min_segment_px]
        energies = [e for e in energies if np.isfinite(e)]
        if energies:
            curvature_bad = float(np.clip(np.mean(energies) / cfg.t6_curvature_scale, 0.0, 1.0))

        weights = {
            "tangent": (cfg.t6_tangent_weight, tangent_bad),
            "curvature": (cfg.t6_curvature_lambda, curvature_bad),
        }
        active = {k: (wt, b) for k, (wt, b) in weights.items() if np.isfinite(b)}
        total = sum(wt for wt, _ in active.values())
        value = -sum(wt * b for wt, b in active.values()) / max(total, 1e-9)

        return TermResult(
            value=value,
            applicable=True,
            diagnostics={
                "n_contacts": len(contacts),
                "n_contact_fits": fits,
                "tangent_bad": tangent_bad,
                "curvature_bad": curvature_bad,
            },
        )
