"""E6 — ground-truth injection diagnostic (SPEC.md §9).

Adds the ground-truth completion to the candidate pool and reports at which
percentile each term (and the combined mode-A score) places it. 0 = best.

This lives in ``eval/`` because it deliberately reads GT; the scoring terms
themselves only ever receive the GT image as *one more candidate*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from counterpart import io as cio
from counterpart.config import Cfg

# importing term modules registers them
from counterpart.score import (  # noqa: F401
    boundary,
    contour,
    frequency,
    rotational,
    symmetry,
    texture,
)
from counterpart.score.base import TERMS, build_score_sample
from counterpart.score.combine import DEFAULT_TERMS, combine_terms


def _descending_percentile(values: np.ndarray, index: int) -> float:
    """Percentile of ``values[index]`` where 0 = best (highest value)."""
    if len(values) <= 1:
        return 0.0
    better = int((values > values[index]).sum())
    return better / (len(values) - 1)


def run_gt_injection(
    cfg: Cfg, split: str, *, experiment: str = "main", limit: int | None = None
) -> dict[str, Any]:
    """Score GT as an extra candidate on samples that already have scores."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]

    terms = DEFAULT_TERMS
    records: list[dict[str, Any]] = []
    skipped = 0
    for sample_id in sample_ids:
        sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
        if not sample.scores_path.exists():
            skipped += 1
            continue
        frame = pd.read_parquet(sample.scores_path).sort_values("idx").reset_index(drop=True)
        terms_sample = [name for name in terms if f"{name}_raw" in frame.columns]
        if not terms_sample:
            skipped += 1
            continue

        # the score context comes from the gt_hidden view; GT is read through the
        # eval-layer view (this module lives in eval/, which may touch ground truth)
        ctx = build_score_sample(sample, cfg)
        gt_sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=False)
        instances = [TERMS[name]() for name in terms_sample]
        prepared = [term.prepare(ctx) for term in instances]
        gt = cio.read_image(gt_sample.original_path)

        gt_raw: dict[str, float] = {}
        gt_applicable: dict[str, bool] = {}
        for term, prep in zip(instances, prepared, strict=True):
            result = term.score(prep, gt)
            gt_raw[term.name] = float(result.value)
            gt_applicable[term.name] = bool(result.applicable)

        raw = {
            name: np.append(frame[f"{name}_raw"].to_numpy(dtype=np.float64), gt_raw[name])
            for name in terms_sample
        }
        applicable = {
            name: np.append(frame[f"{name}_applicable"].to_numpy(dtype=bool), gt_applicable[name])
            for name in terms_sample
        }
        combined, _ = combine_terms(raw, applicable, weights=None)
        gt_index = len(frame)

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "combined_percentile": _descending_percentile(combined, gt_index),
        }
        for name in terms_sample:
            values = raw[name]
            mask = applicable[name] & np.isfinite(values)
            if gt_applicable[name] and np.isfinite(gt_raw[name]):
                pool = values[mask]
                gt_value = gt_raw[name]
                better = int((pool > gt_value).sum())
                row[f"{name}_percentile"] = better / max(len(pool) - 1, 1)
            else:
                row[f"{name}_percentile"] = float("nan")
        records.append(row)

    frame = pd.DataFrame(records)
    summary: dict[str, Any] = {
        "split": split,
        "experiment": experiment,
        "n_images": len(frame),
        "skipped": skipped,
    }
    if not frame.empty:
        summary["mean_combined_percentile"] = round(float(frame["combined_percentile"].mean()), 4)
        summary["frac_gt_top1_combined"] = round(
            float((frame["combined_percentile"] == 0.0).mean()), 4
        )
        summary["frac_gt_top3_combined"] = round(
            float((frame["combined_percentile"] <= (2 / 31)).mean()), 4
        )
        for name in terms:
            column = f"{name}_percentile"
            if column in frame.columns and frame[column].notna().any():
                summary[f"mean_{column}"] = round(float(frame[column].mean()), 4)
        results_dir = runs_dir / experiment / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        cio.save_parquet(results_dir / f"gt_injection_{split}.parquet", frame)
    return summary
