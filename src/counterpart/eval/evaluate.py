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
from counterpart.eval.ranking import percentile_of, selector_summary, spearman_score_vs_lpips
from counterpart.eval.stats import bootstrap_ci
from counterpart.types import Sample

METRIC_COLUMNS = ["lpips", "psnr", "ssim", "delta_e_boundary", "silhouette_iou"]
SELECTOR_NAMES = ["ours_eq", "ours_learned", "clip", "dino", "first", "random_seed"]


def _selector_columns(
    cfg: Cfg,
    sample: Sample,
    metrics: pd.DataFrame,
    lpips_by_idx: dict[int, float],
) -> dict[str, Any]:
    """Per-image selector results, rank correlations and classical-baseline metrics."""
    columns: dict[str, Any] = {}
    lpips_values = metrics["lpips"].to_numpy(dtype=np.float64)
    position = {int(idx): pos for pos, idx in enumerate(metrics["idx"].astype(int))}

    if sample.selection_path.exists():
        selection = cio.load_json(sample.selection_path)
        for name in SELECTOR_NAMES:
            idx = selection.get(name)
            if idx is None or int(idx) not in lpips_by_idx:
                continue
            value = lpips_by_idx[int(idx)]
            columns[f"lpips_sel_{name}"] = value
            columns[f"regret_sel_{name}"] = value - float(lpips_values.min())
            columns[f"pct_sel_{name}"] = percentile_of(lpips_values, position[int(idx)])

    if sample.scores_path.exists():
        scores = pd.read_parquet(sample.scores_path).sort_values("idx")
        lpips_series = metrics.set_index("idx")["lpips"]
        aligned = scores.merge(
            lpips_series.rename("lpips"), left_on="idx", right_index=True, how="inner"
        )
        for column in ("combined_A", "combined_B"):
            if column in aligned.columns and len(aligned) >= 3:
                columns[f"spearman_{column}"] = spearman_score_vs_lpips(
                    aligned[column].to_numpy(dtype=np.float64),
                    aligned["lpips"].to_numpy(dtype=np.float64),
                )
    if sample.baseline_scores_path.exists():
        baselines = pd.read_parquet(sample.baseline_scores_path).sort_values("idx")
        lpips_series = metrics.set_index("idx")["lpips"]
        aligned = baselines.merge(
            lpips_series.rename("lpips"), left_on="idx", right_index=True, how="inner"
        )
        for column in ("clip", "dino"):
            if column in aligned.columns and len(aligned) >= 3 and aligned[column].notna().all():
                columns[f"spearman_{column}"] = spearman_score_vs_lpips(
                    aligned[column].to_numpy(dtype=np.float64),
                    aligned["lpips"].to_numpy(dtype=np.float64),
                )

    baseline_dir = sample.root / "baselines"
    for method in cfg.baselines.classical_methods:
        path = baseline_dir / f"{method}.png"
        if not path.exists():
            continue
        cache = baseline_dir / f"{method}_metrics.json"
        if cache.exists():
            payload = cio.load_json(cache)
        else:
            gt = cio.read_image(sample.original_path)
            candidate = cio.read_image(path)
            payload = reconstruction_metrics(
                gt,
                candidate,
                cio.read_mask(sample.damage_mask_path),
                cio.read_mask(sample.object_mask_path),
                cfg.eval,
                cfg.segment,
            )
            cio.save_json(cache, payload)
        columns[f"lpips_{method}"] = float(payload["lpips"])
        columns[f"psnr_{method}"] = float(payload["psnr"])
        columns[f"iou_{method}"] = float(payload["silhouette_iou"])
    return columns


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
        if expected and expected == set(int(i) for i in df["idx"]):
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
        lpips_by_idx = dict(zip(df["idx"].astype(int), lpips, strict=True))
        selector = selector_summary(lpips)
        oracle_row = df.loc[int(selector["oracle_idx"])]
        row: dict[str, Any] = {
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
        row |= _selector_columns(cfg, sample, df, lpips_by_idx)
        rows.append(row)
        all_lpips.append(lpips)
        if verbose:
            ours = row.get("lpips_ours_eq")
            extra = (
                ""
                if ours is None
                else f" ours_eq={ours:.3f} pct={row.get('pct_ours_eq', float('nan')):.2f}"
            )
            print(
                f"[{i}/{len(sample_ids)}] {sample_id}: "
                f"oracle={selector['lpips_oracle']:.3f} random={selector['lpips_random']:.3f} "
                f"spread={selector['spread']:.3f}{extra}"
            )

    per_image = pd.DataFrame(rows)
    results_dir = runs_dir / experiment / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cio.save_parquet(results_dir / f"per_image_{split}.parquet", per_image)

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

    # --- selection-quality aggregates (E1): per selector LPIPS/regret/percentile/win-rate
    random_values = per_image["lpips_random"].to_numpy()
    for name in SELECTOR_NAMES:
        column = f"lpips_sel_{name}"
        if column not in per_image.columns or not per_image[column].notna().any():
            continue
        values = per_image[column].dropna().to_numpy()
        ci = bootstrap_ci(values, n_boot=n_boot)
        summary[f"mean_lpips_sel_{name}"] = round(ci["mean"], 4)
        summary[f"ci_lo_sel_{name}"] = round(ci["lo"], 4)
        summary[f"ci_hi_sel_{name}"] = round(ci["hi"], 4)
        summary[f"median_lpips_sel_{name}"] = round(float(np.median(values)), 4)
        regret = per_image[f"regret_sel_{name}"].dropna().to_numpy()
        if len(regret):
            summary[f"mean_regret_sel_{name}"] = round(float(regret.mean()), 4)
        pct = per_image[f"pct_sel_{name}"].dropna().to_numpy()
        if len(pct):
            summary[f"mean_pct_sel_{name}"] = round(float(pct.mean()), 4)
        shared = per_image[[column, "lpips_random"]].dropna()
        if len(shared):
            summary[f"winrate_vs_random_{name}"] = round(
                float((shared[column] < shared["lpips_random"]).mean()), 4
            )

    # --- rank correlations (per image, averaged)
    for key in (
        "spearman_combined_A",
        "spearman_combined_B",
        "spearman_clip",
        "spearman_dino",
    ):
        if key in per_image.columns and per_image[key].notna().any():
            summary[f"mean_{key}"] = round(float(per_image[key].mean()), 4)

    # --- classical baselines
    for method in ("telea", "ns", "patchmatch"):
        column = f"lpips_{method}"
        if column in per_image.columns and per_image[column].notna().any():
            ci = bootstrap_ci(per_image[column].dropna().to_numpy(), n_boot=n_boot)
            summary[f"mean_lpips_{method}"] = round(ci["mean"], 4)
            summary[f"ci_lo_{method}"] = round(ci["lo"], 4)
            summary[f"ci_hi_{method}"] = round(ci["hi"], 4)
    return summary
