"""T4 — texture statistics consistency (SPEC.md §5.4 T4).

Reference: LBP histograms, a Gabor energy bank and a Lab colour histogram computed
over the visible object region V (split into SLIC superpixels). The candidate's
reconstructed area is described by the same features and compared to the *nearest*
visible region for each feature family ("does this texture exist anywhere visible?").

Implementation note: feature maps are computed once per image (reference: complete
damaged image; candidate: damage bounding box), and aggregated over the union of
valid patch windows. This keeps both the patch-based statistics of the spec and a
per-candidate cost within the CPU budget (SPEC.md §11).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic

from counterpart.score.base import ScoreSample, register
from counterpart.score.geometry import erode, valid_patch_positions
from counterpart.segment.threshold import rgb_to_lab
from counterpart.types import TermResult

GABOR_FREQUENCIES = (0.05, 0.1, 0.2, 0.4)
GABOR_THETAS = tuple(np.linspace(0, np.pi, 6, endpoint=False))
GABOR_MAX_KSIZE = 41


def _chi2(p: np.ndarray, q: np.ndarray) -> float:
    denominator = p + q + 1e-9
    return float(0.5 * np.sum((p - q) ** 2 / denominator))


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(1.0 - np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0))))


def _valid_patch_union(region: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """Union of all ``patch``-sized windows fully contained in ``region``."""
    union = np.zeros_like(region, dtype=bool)
    for x, y in valid_patch_positions(region, patch, stride):
        union[y : y + patch, x : x + patch] = True
    return union


def _gabor_kernels(frequency: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature pair of Gabor kernels (real / 90°-phase) at the given scale & angle."""
    wavelength = 1.0 / frequency
    sigma = max(0.5 * wavelength, 1.0)
    ksize = min(int(np.ceil(3.0 * sigma)) * 2 + 1, GABOR_MAX_KSIZE)
    if ksize % 2 == 0:
        ksize += 1
    real = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength, 0.5, 0.0, ktype=cv2.CV_32F)
    imag = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, wavelength, 0.5, np.pi / 2.0, ktype=cv2.CV_32F
    )
    return real, imag


def _feature_maps(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel feature maps: two LBP code maps and 24 Gabor magnitude maps.

    OpenCV filter kernels are much cheaper here than skimage's implementation, which
    dominates the runtime otherwise (SPEC.md §11 CPU budget).
    """
    maps: dict[str, np.ndarray] = {
        "lbp8": local_binary_pattern(gray, 8, 1, method="uniform").astype(np.int32),
        "lbp16": local_binary_pattern(gray, 16, 2, method="uniform").astype(np.int32),
    }
    gray32 = gray.astype(np.float32)
    for i, frequency in enumerate(GABOR_FREQUENCIES):
        for j, theta in enumerate(GABOR_THETAS):
            real_kernel, imag_kernel = _gabor_kernels(frequency, float(theta))
            real = cv2.filter2D(gray32, cv2.CV_32F, real_kernel)
            imag = cv2.filter2D(gray32, cv2.CV_32F, imag_kernel)
            maps[f"gabor_{i}_{j}"] = np.hypot(real, imag)
    return maps


def _aggregate(
    maps: dict[str, np.ndarray], lab: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray] | None:
    """Aggregate feature maps over ``mask`` into an LBP / Gabor / colour vector."""
    if int(mask.sum()) < 64:
        return None
    lbp8 = np.bincount(maps["lbp8"][mask], minlength=10)[:10].astype(np.float64)
    lbp16 = np.bincount(maps["lbp16"][mask], minlength=18)[:18].astype(np.float64)
    lbp8 /= max(lbp8.sum(), 1e-9)
    lbp16 /= max(lbp16.sum(), 1e-9)

    gabor_values = []
    for i in range(len(GABOR_FREQUENCIES)):
        for j in range(len(GABOR_THETAS)):
            magnitude = maps[f"gabor_{i}_{j}"][mask]
            gabor_values.extend([float(magnitude.mean()), float(magnitude.std())])

    bins = np.linspace(-80.0, 80.0, 17)
    hist, _, _ = np.histogram2d(lab[..., 1][mask], lab[..., 2][mask], bins=[bins, bins])
    color = hist.ravel().astype(np.float64)
    color /= max(color.sum(), 1e-9)

    return {
        "lbp": np.concatenate([lbp8, lbp16]),
        "gabor": np.asarray(gabor_values, dtype=np.float64),
        "color": color,
    }


@register
class TextureTerm:
    name = "T4"

    # ------------------------------------------------------------------ prepare

    def prepare(self, sample: ScoreSample) -> dict[str, Any]:
        cfg = sample.score_cfg
        patch = cfg.t4_patch_px
        stride = patch // 2

        gray = cv2.cvtColor(sample.damaged, cv2.COLOR_RGB2GRAY)
        maps = _feature_maps(gray)
        lab = sample.damaged_lab

        # SLIC superpixels of the visible object (eroded to avoid Gabor edge artefacts)
        labels = slic(
            sample.damaged,
            n_segments=cfg.t4_superpixel_regions,
            compactness=10.0,
            mask=sample.visible,
            start_label=0,
            channel_axis=-1,
        )
        core = sample.visible & erode(sample.object_mask, 8)

        regions: list[dict[str, np.ndarray]] = []
        for label in np.unique(labels):
            if label < 0:
                continue
            region = (labels == label) & core
            mask = _valid_patch_union(region, patch, stride)
            features = _aggregate(maps, lab, mask)
            if features is not None:
                regions.append(features)

        if not regions:
            mask = _valid_patch_union(core, patch, stride)
            features = _aggregate(maps, lab, mask)
            if features is not None:
                regions.append(features)

        # cap the reference set for cost (nearest-of-N is a min, more regions can only help)
        return {"cfg": cfg, "regions": regions[:16], "damage_mask": sample.damage_mask}

    # -------------------------------------------------------------------- score

    def score(self, prepared: dict[str, Any], candidate: np.ndarray) -> TermResult:
        cfg = prepared["cfg"]
        damage_mask = prepared["damage_mask"]
        regions: list[dict[str, np.ndarray]] = prepared["regions"]
        if not regions:
            return TermResult(
                value=float("nan"),
                applicable=False,
                diagnostics={"reason": "no reference texture regions"},
            )

        ys, xs = np.nonzero(damage_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = candidate[y0:y1, x0:x1]
        crop_mask = damage_mask[y0:y1, x0:x1]

        patch = cfg.t4_patch_px
        union = _valid_patch_union(crop_mask, patch, patch // 2)
        if int(union.sum()) < cfg.t4_min_mask_pixels:
            return TermResult(
                value=float("nan"),
                applicable=False,
                diagnostics={"reason": "damage too small for texture statistics"},
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        maps = _feature_maps(gray)
        features = _aggregate(maps, rgb_to_lab(crop), union)
        if features is None:
            return TermResult(
                value=float("nan"),
                applicable=False,
                diagnostics={"reason": "candidate texture features unavailable"},
            )

        lbp_worst = min(_chi2(features["lbp"], r["lbp"]) for r in regions)
        gabor_worst = min(_chi2(features["gabor"], r["gabor"]) for r in regions)
        color_worst = min(_bhattacharyya(features["color"], r["color"]) for r in regions)

        lbp_bad = float(np.clip(lbp_worst / cfg.t4_lbp_scale, 0.0, 1.0))
        gabor_bad = float(np.clip(gabor_worst / cfg.t4_gabor_scale, 0.0, 1.0))
        color_bad = float(np.clip(color_worst / cfg.t4_color_scale, 0.0, 1.0))

        total = cfg.t4_lbp_weight + cfg.t4_gabor_weight + cfg.t4_color_weight
        badness = (
            cfg.t4_lbp_weight * lbp_bad
            + cfg.t4_gabor_weight * gabor_bad
            + cfg.t4_color_weight * color_bad
        ) / max(total, 1e-9)

        return TermResult(
            value=-badness,
            applicable=True,
            diagnostics={
                "lbp_chi2": lbp_worst,
                "gabor_chi2": gabor_worst,
                "color_bc": color_worst,
                "n_regions": len(regions),
            },
        )
