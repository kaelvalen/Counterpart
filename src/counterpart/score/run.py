"""Scoring orchestration: run terms over a split's candidates (SPEC.md §5.4, CLI `score`).

Ground truth is never touched here: samples are loaded with ``gt_hidden=True`` and
terms only see the damaged input + masks. Results are cached to ``scores.parquet``
per sample (rows = candidates, columns = raw/z/applicable per term + combined_A).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from counterpart import io as cio
from counterpart.config import Cfg

# importing the term modules registers them
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
from counterpart.types import Sample


def score_sample(
    cfg: Cfg,
    experiment: str,
    split: str,
    sample_id: str,
    terms: list[str] | None = None,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Score every candidate of one sample; caches ``scores.parquet``."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample: Sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
    out_path = sample.scores_path

    records = cio.read_jsonl(sample.candidates_manifest_path)
    expected = {int(rec["idx"]) for rec in records}
    if out_path.exists() and not overwrite:
        cached = pd.read_parquet(out_path)
        if expected <= set(int(i) for i in cached["idx"]):
            return cached

    if not records:
        raise ValueError(f"no candidates for {experiment}/{split}/{sample_id}")

    ctx = build_score_sample(sample, cfg)
    term_names = terms or DEFAULT_TERMS
    instances = [TERMS[name]() for name in term_names]
    prepared = [term.prepare(ctx) for term in instances]

    rows: list[dict[str, Any]] = []
    for rec in records:
        idx = int(rec["idx"])
        candidate = cio.read_image(sample.candidate_path(idx))
        row: dict[str, Any] = {"idx": idx}
        for term, prep in zip(instances, prepared, strict=True):
            result = term.score(prep, candidate)
            row[f"{term.name}_raw"] = float(result.value)
            row[f"{term.name}_applicable"] = bool(result.applicable)
            row[f"{term.name}_diag"] = json.dumps(result.diagnostics, default=str)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)

    # mode A combination, stored alongside the raw values
    raw = {name: frame[f"{name}_raw"].to_numpy() for name in term_names}
    applicable = {name: frame[f"{name}_applicable"].to_numpy() for name in term_names}
    combined, z_values = combine_terms(raw, applicable, weights=None)
    frame["combined_A"] = combined
    for name in term_names:
        frame[f"{name}_z"] = z_values[name]

    cio.save_parquet(out_path, frame)
    return frame


def score_sample_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """ProcessPoolExecutor entry point (must stay module-level)."""
    cfg = Cfg.model_validate(payload["cfg"])
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_root = runs_dir / payload["experiment"] / payload["split"] / payload["sample_id"]
    manifest = sample_root / "candidates.jsonl"
    if not manifest.exists():
        return {
            "sample_id": payload["sample_id"],
            "status": "skipped",
            "reason": "no candidates yet",
        }
    try:
        frame = score_sample(
            cfg,
            payload["experiment"],
            payload["split"],
            payload["sample_id"],
            payload["terms"],
            overwrite=payload["overwrite"],
        )
        return {
            "sample_id": payload["sample_id"],
            "status": "ok",
            "n_candidates": int(len(frame)),
        }
    except Exception as exc:  # noqa: BLE001 - report as a status record
        return {
            "sample_id": payload["sample_id"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_score(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    terms: list[str] | None = None,
    limit: int | None = None,
    workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Score all samples of a split (parallel over samples); returns a summary."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no prepared samples under {runs_dir / experiment / split}")

    payloads = [
        {
            "sample_id": sample_id,
            "experiment": experiment,
            "split": split,
            "terms": terms,
            "overwrite": overwrite,
            "cfg": cfg.model_dump(mode="json"),
        }
        for sample_id in sample_ids
    ]

    counts = {"ok": 0, "error": 0}
    errors: list[str] = []
    seconds: list[float] = []
    import time

    if workers <= 1:
        for payload in payloads:
            start = time.perf_counter()
            result = score_sample_worker(payload)
            seconds.append(time.perf_counter() - start)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result["status"] == "error":
                errors.append(f"{result['sample_id']}: {result['error']}")
            if verbose:
                print(
                    f"[{counts['ok'] + counts['error']}/{len(payloads)}] {result['sample_id']}: "
                    f"{result['status']}"
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(score_sample_worker, payloads):
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                if result["status"] == "error":
                    errors.append(f"{result['sample_id']}: {result['error']}")
                if verbose:
                    done = counts["ok"] + counts["error"]
                    print(f"[{done}/{len(payloads)}] {result['sample_id']}: {result['status']}")

    summary: dict[str, Any] = {
        "split": split,
        "experiment": experiment,
        "samples": len(sample_ids),
        **counts,
    }
    if seconds:
        summary["mean_seconds_per_sample"] = round(float(np.mean(seconds)), 2)
    if errors:
        summary["errors"] = errors[:10]
    return summary
