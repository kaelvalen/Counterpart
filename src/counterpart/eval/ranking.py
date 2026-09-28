"""Selection-quality metrics and built-in selectors (SPEC.md §7, §8.2).

Candidate-level ground-truth metrics (LPIPS) come from ``eval.metrics``; this module
turns an image's candidate pool into selector scores and ranking statistics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def selector_summary(lpips: np.ndarray) -> dict[str, float | int]:
    """Per-image values of the built-in selectors (lower LPIPS = better).

    ``random`` is the *expected value* of a random pick (mean over candidates), so it
    carries no single-draw noise (SPEC.md §7).
    """
    return {
        "lpips_first": float(lpips[0]),
        "lpips_random": float(lpips.mean()),
        "lpips_oracle": float(lpips.min()),
        "lpips_worst": float(lpips.max()),
        "lpips_median": float(np.median(lpips)),
        "spread": float(lpips.std()),
        "oracle_idx": int(lpips.argmin()),
        "worst_idx": int(lpips.argmax()),
        "n_candidates": int(len(lpips)),
    }


def percentile_of(values: np.ndarray, idx: int) -> float:
    """Rank of ``idx`` in ``values``: 0 = best (lowest), 1 = worst; random ≈ 0.5."""
    if len(values) <= 1:
        return 0.0
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    rank = int(np.where(order == idx)[0][0])
    return rank / (len(values) - 1)


def spearman_score_vs_lpips(scores: np.ndarray, lpips: np.ndarray) -> float:
    """Spearman ρ between a score (higher = better) and -LPIPS, per image."""
    if len(scores) < 3:
        return float("nan")
    stat = sps.spearmanr(scores, -np.asarray(lpips, dtype=np.float64)).statistic
    return float(stat)


def top1_regret(lpips: np.ndarray, picked_idx: int) -> float:
    """LPIPS(picked) - LPIPS(oracle)."""
    return float(lpips[picked_idx] - lpips.min())


def win_rate(selector_lpips: np.ndarray, random_lpips: np.ndarray) -> float:
    """Fraction of images where the selector beats the random expectation."""
    diff = np.asarray(selector_lpips) - np.asarray(random_lpips)
    return float((diff < 0).mean())
