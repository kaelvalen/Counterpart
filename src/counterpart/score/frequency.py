"""T5 — frequency profile consistency (SPEC.md §5.4 T5).

Diffusion outputs are often over-smooth or carry spurious high-frequency energy.
Reference: radial average power spectrum over visible-object patches. Candidate:
same statistic over patches inside the damage region. Both are summarised by the
log-log spectral slope and the high-frequency energy ratio (f > cutoff).

Only patches fully inside the respective region count; cracks are too thin, so the
term abstains there (a documented, expected outcome — SPEC.md §6.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from counterpart.score.base import ScoreSample, register
from counterpart.score.geometry import valid_patch_positions
from counterpart.types import TermResult


def _hann_window(patch: int) -> np.ndarray:
    one_d = np.hanning(patch)
    return np.outer(one_d, one_d)


def _radial_profile(patch_values: np.ndarray, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed FFT -> radially averaged power spectrum and normalised frequency axis."""
    spectrum = np.fft.fftshift(np.fft.fft2(patch_values * window))
    power = (spectrum.real**2 + spectrum.imag**2) / patch_values.size
    patch = patch_values.shape[0]
    cy = cx = patch // 2
    yy, xx = np.mgrid[0:patch, 0:patch]
    radius = np.hypot(yy - cy, xx - cx)
    max_radius = patch / 2
    bins = np.arange(0, max_radius + 1)
    profile = np.zeros_like(bins, dtype=np.float64)
    for i in range(len(bins) - 1):
        ring = (radius >= bins[i]) & (radius < bins[i + 1])
        if ring.any():
            profile[i] = power[ring].mean()
    frequencies = bins / max_radius
    return profile, frequencies


def _spectral_summary(
    patch_values: np.ndarray, window: np.ndarray, cutoff: float
) -> tuple[float, float]:
    """Log-log slope (excluding DC) and the high-frequency energy ratio."""
    profile, frequencies = _radial_profile(patch_values, window)
    valid = (frequencies > 0) & (profile > 0)
    if valid.sum() >= 4:
        log_f = np.log(frequencies[valid])
        log_p = np.log(profile[valid])
        slope = float(np.polyfit(log_f, log_p, 1)[0])
    else:
        slope = float("nan")
    total = profile[1:].sum()
    high = profile[frequencies > cutoff].sum()
    ratio = float(high / total) if total > 0 else float("nan")
    return slope, ratio


def _patch_summary(
    image_gray: np.ndarray, positions: list[tuple[int, int]], patch: int, cutoff: float
) -> tuple[float, float] | None:
    if not positions:
        return None
    window = _hann_window(patch)
    slopes, ratios = [], []
    for x, y in positions:
        slope, ratio = _spectral_summary(
            image_gray[y : y + patch, x : x + patch].astype(np.float64), window, cutoff
        )
        if np.isfinite(slope) and np.isfinite(ratio):
            slopes.append(slope)
            ratios.append(ratio)
    if not slopes:
        return None
    return float(np.median(slopes)), float(np.median(ratios))


@register
class FrequencyTerm:
    name = "T5"

    def prepare(self, sample: ScoreSample) -> dict[str, Any]:
        cfg = sample.score_cfg
        patch = cfg.t5_patch_px
        stride = patch // 2
        gray = np.asarray(np.dot(sample.damaged[..., :3], [0.299, 0.587, 0.114]), dtype=np.float64)
        positions = valid_patch_positions(sample.visible, patch, stride)[:128]
        summary = _patch_summary(gray, positions, patch, cfg.t5_high_freq_cutoff)
        if summary is None:
            return {"cfg": cfg, "reference": None, "damage_mask": sample.damage_mask}
        return {"cfg": cfg, "reference": summary, "damage_mask": sample.damage_mask}

    def score(self, prepared: dict[str, Any], candidate: np.ndarray) -> TermResult:
        cfg = prepared["cfg"]
        reference = prepared["reference"]
        damage_mask = prepared["damage_mask"]
        if reference is None:
            return TermResult(
                value=float("nan"),
                applicable=False,
                diagnostics={"reason": "no reference patches"},
            )

        gray = np.asarray(np.dot(candidate[..., :3], [0.299, 0.587, 0.114]), dtype=np.float64)
        patch = cfg.t5_patch_px
        stride = patch // 2
        positions = valid_patch_positions(damage_mask, patch, stride)[:128]
        summary = _patch_summary(gray, positions, patch, cfg.t5_high_freq_cutoff)
        if summary is None or len(positions) < cfg.t5_min_patches:
            return TermResult(
                value=float("nan"),
                applicable=False,
                diagnostics={"reason": "damage too small for frequency patches"},
            )

        ref_slope, ref_ratio = reference
        slope, ratio = summary
        slope_bad = float(np.clip(abs(slope - ref_slope) / cfg.t5_slope_scale, 0.0, 1.0))
        hf_bad = float(np.clip(abs(ratio - ref_ratio) / cfg.t5_hf_scale, 0.0, 1.0))
        value = -0.5 * (slope_bad + hf_bad)
        return TermResult(
            value=value,
            applicable=True,
            diagnostics={
                "slope": slope,
                "slope_ref": ref_slope,
                "hf_ratio": ratio,
                "hf_ratio_ref": ref_ratio,
                "n_patches": len(positions),
            },
        )
