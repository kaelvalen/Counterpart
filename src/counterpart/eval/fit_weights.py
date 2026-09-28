"""Mode-B weight fitting on the train split (SPEC.md §5.4, §7).

Reads the cached per-term z-scores plus ground-truth LPIPS (this module lives in
``eval/`` — the only place besides ``combine.py`` allowed to touch GT, and only for
the *train* split), fits pairwise logistic weights and stores them in
``runs/<experiment>/results/weights_mode_b.json`` for the ``ours_learned`` selector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.score.combine import DEFAULT_TERMS, pairwise_logistic_fit, save_weights


def fit_mode_b(
    cfg: Cfg,
    *,
    experiment: str = "main",
    train_split: str = "train",
    terms: list[str] | None = None,
    l2: float = 1.0,
    iters: int = 400,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fit mode-B weights on a train split; returns a summary incl. the weights."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, train_split)
    if limit is not None:
        sample_ids = sample_ids[:limit]

    feature_rows: list[np.ndarray] = []
    lpips_rows: list[np.ndarray] = []
    group_rows: list[np.ndarray] = []
    term_names: list[str] | None = terms
    used = 0

    for group_index, sample_id in enumerate(sample_ids):
        sample = cio.load_sample(runs_dir, experiment, train_split, sample_id)
        scores_path = sample.scores_path
        metrics_path = sample.root / "metrics.parquet"
        if not scores_path.exists() or not metrics_path.exists():
            continue
        scores = pd.read_parquet(scores_path).sort_values("idx")
        metrics = pd.read_parquet(metrics_path).sort_values("idx")

        aligned = scores.merge(metrics[["idx", "lpips"]], on="idx", how="inner")
        if len(aligned) < 8:
            continue

        if term_names is None:
            term_names = [
                name for name in (terms or DEFAULT_TERMS) if f"{name}_z" in aligned.columns
            ]
        missing = [name for name in term_names if f"{name}_z" not in aligned.columns]
        if missing:
            continue

        features = np.stack(
            [aligned[f"{name}_z"].to_numpy(dtype=np.float64) for name in term_names], axis=1
        )
        lpips = aligned["lpips"].to_numpy(dtype=np.float64)
        feature_rows.append(features)
        lpips_rows.append(lpips)
        group_rows.append(np.full(len(aligned), group_index))
        used += 1

    if not feature_rows or term_names is None:
        raise ValueError(
            f"no samples with scores+metrics under {runs_dir / experiment / train_split}; "
            "run `score` and `evaluate` first"
        )

    features = np.concatenate(feature_rows)
    lpips = np.concatenate(lpips_rows)
    groups = np.concatenate(group_rows)

    weights_vector = pairwise_logistic_fit(features, lpips, groups=groups, l2=l2, iters=iters)
    weights = {name: float(w) for name, w in zip(term_names, weights_vector, strict=True)}

    weights_path = runs_dir / experiment / "results" / "weights_mode_b.json"
    save_weights(
        weights_path,
        weights,
        meta={
            "train_split": train_split,
            "experiment": experiment,
            "n_samples": used,
            "n_candidates": int(len(features)),
            "l2": l2,
            "iters": iters,
        },
    )
    return {
        "experiment": experiment,
        "train_split": train_split,
        "n_samples": used,
        "n_candidates": int(len(features)),
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "path": str(weights_path),
    }
