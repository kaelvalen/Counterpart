"""Pixel-level uncertainty from candidate dispersion (SPEC.md §5.5).

``U_pix``: per-pixel standard deviation of the candidates' CIELAB values, averaged
over channels and zero outside the damage mask. The full pool is used, plus a top-K
variant (default 8, ranked by combined_A) for the report comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.segment.threshold import rgb_to_lab


def _dispersion(candidates: list[np.ndarray]) -> np.ndarray:
    """Mean-over-channel std of Lab values across candidates (float32 HxW)."""
    h, w = candidates[0].shape[:2]
    total = np.zeros((h, w, 3), dtype=np.float64)
    total_sq = np.zeros((h, w, 3), dtype=np.float64)
    for candidate in candidates:
        lab = rgb_to_lab(candidate).astype(np.float64)
        total += lab
        total_sq += lab * lab
    n = len(candidates)
    mean = total / n
    variance = np.maximum(total_sq / n - mean * mean, 0.0)
    return np.sqrt(variance).mean(axis=2).astype(np.float32)


def compute_uncertainty_sample(
    cfg: Cfg, experiment: str, split: str, sample_id: str, *, overwrite: bool = False
) -> dict[str, Any]:
    """Compute and cache U_pix (all candidates + top-K variant) for one sample."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
    out_path = sample.uncertainty_path
    top_path = sample.root / f"uncertainty_top{cfg.select.uncertainty_top_k_variant}.npy"
    if out_path.exists() and top_path.exists() and not overwrite:
        uncertainty = np.load(out_path)
        return {
            "sample_id": sample_id,
            "mean_u": float(uncertainty.mean()),
            "cached": True,
        }

    scores = pd.read_parquet(sample.scores_path)
    records = cio.read_jsonl(sample.candidates_manifest_path)
    if not records:
        raise ValueError(f"no candidates for {sample_id}")

    damage_mask = cio.read_mask(sample.damage_mask_path)
    ordered = scores.sort_values("combined_A", ascending=False)["idx"].tolist()

    all_candidates = [cio.read_image(sample.candidate_path(int(idx))) for idx in ordered]
    uncertainty = _dispersion(all_candidates)
    uncertainty[~damage_mask] = 0.0

    top_k = min(cfg.select.uncertainty_top_k_variant, len(all_candidates))
    uncertainty_top = _dispersion(all_candidates[:top_k])
    uncertainty_top[~damage_mask] = 0.0

    np.save(sample.uncertainty_path, uncertainty)
    np.save(top_path, uncertainty_top)
    return {
        "sample_id": sample_id,
        "mean_u": float(uncertainty.mean()),
        "mean_u_top": float(uncertainty_top.mean()),
        "cached": False,
    }


def uncertainty_heatmap(uncertainty: np.ndarray, damage_mask: np.ndarray) -> np.ndarray:
    """Render U_pix as an RGB heatmap restricted to the damage mask (for panels)."""
    values = uncertainty.astype(np.float32).copy()
    values[~damage_mask] = np.nan
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    normalized = np.zeros_like(values)
    normalized[np.isfinite(values)] = values[np.isfinite(values)] / max(vmax, 1e-9)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~damage_mask] = (245, 245, 245)
    return colored


def run_uncertainty(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    limit: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute uncertainty maps over a split (serial; memory-frugal accumulation)."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no samples under {runs_dir / experiment / split}")

    means = []
    for i, sample_id in enumerate(sample_ids, start=1):
        result = compute_uncertainty_sample(cfg, experiment, split, sample_id, overwrite=overwrite)
        means.append(result["mean_u"])
        if verbose:
            print(f"[{i}/{len(sample_ids)}] {sample_id}: mean U = {result['mean_u']:.2f}")
    return {
        "split": split,
        "experiment": experiment,
        "samples": len(sample_ids),
        "mean_u": round(float(np.mean(means)), 3) if means else 0.0,
    }
