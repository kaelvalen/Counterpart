"""Statistical helpers (SPEC.md §8.4): bootstrap CIs and paired tests with Holm correction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as sps


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI of the mean (SPEC.md §8.4)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def median_ci(
    values: Sequence[float] | np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    medians = np.median(arr[idx], axis=1)
    lo, hi = np.percentile(medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"median": float(np.median(arr)), "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def wilcoxon_holm(
    reference: dict[str, np.ndarray], comparison_key: str, candidates: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Paired Wilcoxon signed-rank of ``candidates[k]`` vs ``reference[comparison_key]``.

    ``candidates`` maps selector name -> per-image values; returns p-values plus the
    Holm-corrected decision at alpha=0.05.
    """
    ref = np.asarray(reference[comparison_key], dtype=np.float64)
    names = [name for name in candidates if name != comparison_key]
    p_values: dict[str, float] = {}
    for name in names:
        values = np.asarray(candidates[name], dtype=np.float64)
        mask = np.isfinite(ref) & np.isfinite(values) & (ref != values)
        if mask.sum() < 5:
            p_values[name] = float("nan")
            continue
        p_values[name] = float(sps.wilcoxon(values[mask], ref[mask]).pvalue)

    # Holm correction
    order = sorted(names, key=lambda name: (np.isnan(p_values[name]), p_values[name]))
    m = len(order)
    out: dict[str, dict[str, float]] = {}
    previous = 0.0
    for rank, name in enumerate(order):
        p = p_values[name]
        adjusted = min(1.0, (m - rank) * p) if np.isfinite(p) else float("nan")
        adjusted = max(adjusted, previous)  # enforce monotonicity
        previous = adjusted
        out[name] = {"p": p, "p_holm": adjusted, "significant": bool(adjusted < 0.05)}
    return out
