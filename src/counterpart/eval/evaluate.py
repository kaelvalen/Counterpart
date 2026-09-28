"""GT evaluation orchestration (SPEC.md §8, §11 Faz 2).

Per sample: candidate-level GT metrics cached to ``metrics.parquet``.
Aggregated: ``runs/<exp>/results/per_image_<split>.parquet`` and ``summary_<split>.csv``.
For the gonogo split the E0 decision rule is applied and figures are produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.eval.metrics import reconstruction_metrics
from counterpart.eval.ranking import selector_summary
from counterpart.eval.stats import bootstrap_ci

METRIC_COLUMNS = ["lpips", "psnr", "ssim", "delta_e_boundary", "silhouette_iou"]


def evaluate_sample(
    cfg: Cfg, experiment: str, split: str, sample_id: str, *, overwrite: bool = False
) -> pd.DataFrame:
    """Candidate-level GT metrics for one sample (cached in the sample dir)."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample = cio.load_sample(runs_dir, experiment, split, sample_id)  # GT is allowed here
    cache = sample.root / "metrics.parquet"
    records = cio.read_jsonl(sample.candidates_manifest_path)
    expected = {int(rec["idx"]) for rec in records}

    if cache.exists() and not overwrite:
        df = pd.read_parquet(cache)
        if expected <= set(int(i) for i in df["idx"]):
            return df

    gt = cio.read_image(sample.original_path)
    damage_mask = cio.read_mask(sample.damage_mask_path)
    gt_object_mask = cio.read_mask(sample.object_mask_path)

    rows = []
    for rec in records:
        idx = int(rec["idx"])
        candidate = cio.read_image(sample.candidate_path(idx))
        metrics = reconstruction_metrics(
            gt, candidate, damage_mask, gt_object_mask, cfg.eval, cfg.segment
        )
        rows.append({"idx": idx, **metrics})
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    cio.save_parquet(cache, df)
    return df


def run_evaluate(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    limit: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate all prepared+generated samples of a split; returns the summary."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no samples under {runs_dir / experiment / split}")

    rows: list[dict[str, Any]] = []
    all_lpips: list[np.ndarray] = []
    for i, sample_id in enumerate(sample_ids, start=1):
        sample = cio.load_sample(runs_dir, experiment, split, sample_id)
        manifest = cio.read_jsonl(sample.candidates_manifest_path)
        if not manifest:
            if verbose:
                print(f"[{i}/{len(sample_ids)}] {sample_id}: no candidates, skipped")
            continue
        df = evaluate_sample(cfg, experiment, split, sample_id, overwrite=overwrite)
        lpips = df["lpips"].to_numpy(dtype=np.float64)
        selector = selector_summary(lpips)
        oracle_row = df.loc[int(selector["oracle_idx"])]
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "product_type": sample.meta.get("product_type"),
                "damage_type": (sample.meta.get("damage") or {}).get("damage_type"),
                "area_frac": (sample.meta.get("damage") or {}).get("area_frac"),
                **selector,
                "oracle_psnr": float(oracle_row["psnr"]),
                "oracle_ssim": float(oracle_row["ssim"]),
                "oracle_delta_e": float(oracle_row["delta_e_boundary"]),
                "oracle_iou": float(oracle_row["silhouette_iou"]),
                "mean_psnr": float(df["psnr"].mean()),
                "mean_ssim": float(df["ssim"].mean()),
            }
        )
        all_lpips.append(lpips)
        if verbose:
            print(
                f"[{i}/{len(sample_ids)}] {sample_id}: "
                f"oracle={selector['lpips_oracle']:.3f} random={selector['lpips_random']:.3f} "
                f"spread={selector['spread']:.3f}"
            )

    per_image = pd.DataFrame(rows)
    results_dir = runs_dir / experiment / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cio.save_parquet(per_image, results_dir / f"per_image_{split}.parquet")

    summary = _summarize(per_image, split, n_boot=cfg.eval.pairwise_bootstrap)
    pd.DataFrame([summary]).to_csv(results_dir / f"summary_{split}.csv", index=False)

    if split == "gonogo" and not per_image.empty:
        from counterpart.viz.figures import e0_figures

        figures = e0_figures(per_image, all_lpips, results_dir)
        summary["figures"] = [str(p) for p in figures]

    return summary


def _summarize(per_image: pd.DataFrame, split: str, n_boot: int = 1000) -> dict[str, Any]:
    """Aggregate per-image rows into the split-level summary + E0 decision."""
    summary: dict[str, Any] = {"split": split, "n_images": int(len(per_image))}
    if per_image.empty:
        return summary

    for key in ["lpips_oracle", "lpips_first", "lpips_random", "lpips_worst", "spread"]:
        ci = bootstrap_ci(per_image[key].to_numpy(), n_boot=n_boot)
        summary[f"mean_{key}"] = round(ci["mean"], 4)
        summary[f"ci_lo_{key}"] = round(ci["lo"], 4)
        summary[f"ci_hi_{key}"] = round(ci["hi"], 4)

    summary["gap"] = round(summary["mean_lpips_random"] - summary["mean_lpips_oracle"], 4)
    summary["oracle_rel"] = round(summary["mean_lpips_oracle"] / summary["mean_lpips_random"], 4)
    summary["gap_over_half_spread"] = round(
        summary["gap"] / max(summary["mean_spread"] / 2, 1e-9), 3
    )

    # E0 decision rule (SPEC.md §9)
    if summary["oracle_rel"] <= 0.80 and summary["gap"] > summary["mean_spread"] / 2:
        decision = "continue"
    elif summary["oracle_rel"] <= 0.90:
        decision = "borderline (increase diversity and rerun E0)"
    else:
        decision = "stop (report to Kael; consider H2-focused pivot)"
    summary["e0_decision"] = decision

    # E3 material: does the gap shrink as N grows? (analysis only, computed in Faz 5)
    summary["median_spread"] = round(float(per_image["spread"].median()), 4)
    summary["mean_oracle_psnr"] = round(float(per_image["oracle_psnr"].mean()), 3)
    summary["mean_oracle_ssim"] = round(float(per_image["oracle_ssim"].mean()), 4)
    summary["mean_oracle_delta_e"] = round(float(per_image["oracle_delta_e"].mean()), 3)
    summary["mean_oracle_iou"] = round(float(per_image["oracle_iou"].mean()), 4)
    return summary
