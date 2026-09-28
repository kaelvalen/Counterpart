"""Selection: one pick per selector, written to ``selection.json`` (SPEC.md §5.5, §7).

Selectors over the same candidate pool:

* ``ours_eq``   — max combined_A (equal-weight classical terms)
* ``ours_learned`` — max combined_B when mode-B weights exist (train-fitted)
* ``clip`` / ``dino`` — max baseline score
* ``first``     — idx 0 (no selection)
* ``random_seed`` — a deterministic random pick (for panels); the *expected* value of
  random selection is evaluated analytically in ``eval`` (no single-draw noise)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from counterpart import io as cio
from counterpart.config import Cfg


def _mode_b_weights_path(cfg: Cfg, experiment: str) -> Path:
    return Path(cfg.project.paths.runs_dir) / experiment / "results" / "weights_mode_b.json"


def combined_b(frame: pd.DataFrame, weights: dict[str, float], terms: list[str]) -> np.ndarray:
    """Mode-B combination from stored z-columns and learned weights."""
    combined = np.zeros(len(frame), dtype=np.float64)
    total = 0.0
    for name in terms:
        column = f"{name}_z"
        if column not in frame.columns:
            continue
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        combined += weight * frame[column].to_numpy(dtype=np.float64)
        total += weight
    return combined / total if total > 0 else combined


def _random_pick(sample_id: str, n: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % max(n, 1)


def select_sample(
    cfg: Cfg, experiment: str, split: str, sample_id: str, *, overwrite: bool = False
) -> dict[str, Any]:
    """Compute the selector picks for one sample (resumable)."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
    out_path = sample.selection_path

    scores = pd.read_parquet(sample.scores_path)
    if not (sample.baseline_scores_path).exists():
        raise FileNotFoundError(
            f"baseline scores missing for {sample_id}; run `counterpart baselines` first"
        )
    baselines = pd.read_parquet(sample.baseline_scores_path)
    frame = scores.merge(baselines, on="idx", how="left")

    picks: dict[str, Any] = {}
    picks["ours_eq"] = int(frame.loc[frame["combined_A"].idxmax(), "idx"])

    weights_path = _mode_b_weights_path(cfg, experiment)
    if weights_path.exists():
        payload = cio.load_json(weights_path)
        weights = payload.get("weights", {})
        terms = [c[: -len("_z")] for c in frame.columns if c.endswith("_z")]
        frame = frame.copy()
        frame["combined_B"] = combined_b(frame, weights, terms)
        picks["ours_learned"] = int(frame.loc[frame["combined_B"].idxmax(), "idx"])
    else:
        picks["ours_learned"] = None

    for name in ("clip", "dino"):
        if name in frame.columns and frame[name].notna().any():
            picks[name] = int(frame.loc[frame[name].idxmax(), "idx"])
        else:
            picks[name] = None

    picks["first"] = int(frame["idx"].min())
    picks["random_seed"] = _random_pick(sample_id, len(frame), cfg.select.random_seed)
    picks["n_candidates"] = int(len(frame))

    if out_path.exists() and not overwrite:
        cached = cio.load_json(out_path)
        if cached.get("n_candidates") == picks["n_candidates"]:
            return cached

    cio.save_json(out_path, picks)
    return picks


def run_select(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    limit: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run selection over a split; returns a summary."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no samples under {runs_dir / experiment / split}")

    from collections import Counter

    counts: Counter[str] = Counter()
    for i, sample_id in enumerate(sample_ids, start=1):
        sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
        if not sample.scores_path.exists():
            counts["skipped"] += 1
            continue
        picks = select_sample(cfg, experiment, split, sample_id, overwrite=overwrite)
        counts["ok"] += 1
        for name, value in picks.items():
            if isinstance(value, int) and name != "n_candidates":
                counts[f"{name}_picked"] += 1
        if verbose:
            print(
                f"[{i}/{len(sample_ids)}] {sample_id}: "
                f"ours_eq={picks['ours_eq']} clip={picks['clip']} dino={picks['dino']}"
            )

    return {"split": split, "experiment": experiment, "samples": len(sample_ids), **counts}
