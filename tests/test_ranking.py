"""Selection-metric tests: selectors, percentiles, regret, rank correlation."""

from __future__ import annotations

import numpy as np
import pytest

from counterpart.eval.ranking import (
    percentile_of,
    selector_summary,
    spearman_score_vs_lpips,
    top1_regret,
    win_rate,
)


def test_selector_summary_values() -> None:
    lpips = np.array([0.5, 0.2, 0.9, 0.4])
    s = selector_summary(lpips)
    assert s["lpips_first"] == pytest.approx(0.5)
    assert s["lpips_oracle"] == pytest.approx(0.2)
    assert s["lpips_worst"] == pytest.approx(0.9)
    assert s["lpips_random"] == pytest.approx(0.5)
    assert s["oracle_idx"] == 1
    assert s["worst_idx"] == 2


def test_percentile_of_bounds() -> None:
    values = np.array([0.5, 0.2, 0.9, 0.4])
    assert percentile_of(values, 1) == 0.0  # best
    assert percentile_of(values, 2) == 1.0  # worst
    assert percentile_of(values, 0) == pytest.approx(2 / 3)


def test_top1_regret() -> None:
    lpips = np.array([0.5, 0.2, 0.9])
    assert top1_regret(lpips, 1) == pytest.approx(0.0)
    assert top1_regret(lpips, 0) == pytest.approx(0.3)


def test_win_rate() -> None:
    selector = np.array([0.1, 0.4, 0.2])
    random = np.array([0.3, 0.3, 0.3])
    assert win_rate(selector, random) == pytest.approx(2 / 3)


def test_spearman_sign() -> None:
    lpips = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    perfect_score = -lpips
    assert spearman_score_vs_lpips(perfect_score, lpips) == pytest.approx(1.0)
    assert spearman_score_vs_lpips(lpips, lpips) == pytest.approx(-1.0)
    assert np.isnan(spearman_score_vs_lpips(np.array([1.0, 2.0]), lpips[:2]))
