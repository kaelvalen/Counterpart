"""T1 — boundary continuity around the damage seam (SPEC.md §5.4 T1).

Sub-terms: seam colour difference (Lab ΔE across ∂M), gradient continuity (magnitude
+ orientation) and the dangling-edge ratio (edges reaching ∂M from the visible side
without a continuation inside the generated region). Higher value = better.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from counterpart.score.base import ScoreSample, boundary_normals, register, sample_band
from counterpart.score.geometry import dilate
from counterpart.segment.threshold import delta_e76, rgb_to_lab
from counterpart.types import TermResult


def _gray_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, np.hypot(gx, gy)


@register
class BoundaryTerm:
    name = "T1"

    # ------------------------------------------------------------------ prepare

    def prepare(self, sample: ScoreSample) -> dict[str, Any]:
        cfg = sample.score_cfg
        points, normals = boundary_normals(sample, cfg.t1_sample_step_px)
        depths = np.arange(1, cfg.t1_seam_depth_px + 1, dtype=np.float64)

        # visible-side reference values (from the damaged input)
        out_colors = sample_band(sample.damaged_lab, points, normals, depths).mean(axis=1)
        gx, gy, mag = _gray_gradients(sample.damaged)
        out_mag = sample_band(mag, points, normals, depths).mean(axis=1)
        out_gx = sample_band(gx, points, normals, depths).mean(axis=1)
        out_gy = sample_band(gy, points, normals, depths).mean(axis=1)

        # boundary contact points where a visible Canny edge arrives *perpendicularly*
        # (edges parallel to ∂M are the damage boundary itself, not dangling edges)
        edge_out = cv2.Canny(sample.damaged, cfg.t1_canny_low, cfg.t1_canny_high).astype(bool)
        edge_dil = dilate(edge_out, cfg.t1_contact_radius_px)
        h, w = sample.shape
        contact = np.zeros(len(points), dtype=bool)
        for i, (x, y) in enumerate(points):
            ix, iy = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
            if not edge_dil[iy, ix]:
                continue
            gx_p, gy_p = float(gx[iy, ix]), float(gy[iy, ix])
            norm = float(np.hypot(gx_p, gy_p))
            if norm < 1e-6:
                continue
            nx, ny = normals[i]
            alignment = abs((gx_p * nx + gy_p * ny) / norm)
            if alignment < cfg.t1_contact_max_alignment:
                contact[i] = True

        return {
            "cfg": cfg,
            "points": points,
            "normals": normals,
            "depths": depths,
            "out_colors": out_colors,
            "out_mag": out_mag,
            "out_gx": out_gx,
            "out_gy": out_gy,
            "contact": contact,
        }

    # -------------------------------------------------------------------- score

    def score(self, prepared: dict[str, Any], candidate: np.ndarray) -> TermResult:
        cfg = prepared["cfg"]
        points = prepared["points"]
        normals = prepared["normals"]
        depths = prepared["depths"]

        if len(points) < 8:
            return TermResult(
                value=float("nan"), applicable=False, diagnostics={"reason": "few boundary points"}
            )

        # --- seam colour difference (Lab)
        cand_lab = rgb_to_lab(candidate)
        in_colors = sample_band(cand_lab, points, normals, -depths).mean(axis=1)
        seam_delta_e = delta_e76(in_colors, prepared["out_colors"])
        seam_bad = float(np.clip(np.median(seam_delta_e) / cfg.t1_seam_scale, 0.0, 1.0))

        # --- gradient continuity
        gx, gy, mag = _gray_gradients(candidate)
        in_mag = sample_band(mag, points, normals, -depths).mean(axis=1)
        in_gx = sample_band(gx, points, normals, -depths).mean(axis=1)
        in_gy = sample_band(gy, points, normals, -depths).mean(axis=1)
        mag_bad = float(
            np.clip(np.mean(np.abs(in_mag - prepared["out_mag"])) / cfg.t1_gradient_scale, 0.0, 1.0)
        )
        dot = in_gx * prepared["out_gx"] + in_gy * prepared["out_gy"]
        norm_prod = np.maximum(
            np.hypot(in_gx, in_gy) * np.hypot(prepared["out_gx"], prepared["out_gy"]), 1e-6
        )
        cos_sim = np.clip(dot / norm_prod, 0.0, 1.0)
        angle_bad = float(np.clip(np.mean(1.0 - cos_sim), 0.0, 1.0))
        gradient_bad = 0.5 * mag_bad + 0.5 * angle_bad

        # --- dangling edges (contacts from the visible side without continuation inside)
        contact = prepared["contact"]
        dangling_bad = float("nan")
        n_contacts = int(contact.sum())
        if n_contacts >= cfg.t1_min_contacts:
            edge_cand = cv2.Canny(candidate, cfg.t1_canny_low, cfg.t1_canny_high).astype(bool)
            h, w = candidate.shape[:2]
            perp = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
            continuations = np.zeros(n_contacts, dtype=bool)
            offsets = np.arange(-2, 3, dtype=np.float64)
            for i, cidx in enumerate(np.nonzero(contact)[0]):
                found = False
                for depth in depths:
                    base = points[cidx] + normals[cidx] * depth
                    for off in offsets:
                        x, y = base + perp[cidx] * off
                        ix, iy = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
                        if edge_cand[iy, ix]:
                            found = True
                            break
                    if found:
                        break
                continuations[i] = found
            dangling_bad = float(1.0 - continuations.mean())

        # --- combine active sub-terms
        weights = {
            "seam": (cfg.t1_seam_weight, seam_bad),
            "gradient": (cfg.t1_gradient_weight, gradient_bad),
            "dangling": (cfg.t1_dangling_weight, dangling_bad),
        }
        active = {k: (w, b) for k, (w, b) in weights.items() if np.isfinite(b)}
        total_weight = sum(w for w, _ in active.values())
        badness = sum(w * b for w, b in active.values()) / max(total_weight, 1e-9)
        return TermResult(
            value=-badness,
            applicable=True,
            diagnostics={
                "seam_delta_e_median": float(np.median(seam_delta_e)),
                "seam_bad": seam_bad,
                "mag_bad": mag_bad,
                "angle_bad": angle_bad,
                "n_contacts": n_contacts,
                "dangling_bad": dangling_bad,
            },
        )
