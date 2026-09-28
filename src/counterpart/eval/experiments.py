"""Cache-only experiment analyses (SPEC.md §9): E2, E3, E4, E5.

All four read the caches produced by score/evaluate/select — no GPU, no new
candidates. Outputs: CSV tables under ``runs/<exp>/results/`` plus E2/E3 figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.eval.stats import bootstrap_ci
from counterpart.score.combine import DEFAULT_TERMS


def _load_rows(cfg: Cfg, experiment: str, split: str) -> list[dict[str, Any]]:
    """One entry per sample with scores+metrics merged on idx."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    rows = []
    for sample_id in cio.iter_sample_ids(runs_dir, experiment, split):
        sample = cio.load_sample(runs_dir, experiment, split, sample_id)
        if not sample.scores_path.exists():
            continue
        metrics_path = sample.root / "metrics.parquet"
        if not metrics_path.exists():
            continue
        scores = pd.read_parquet(sample.scores_path).sort_values("idx")
        metrics = pd.read_parquet(metrics_path).sort_values("idx")
        merged = scores.merge(metrics[["idx", "lpips"]], on="idx", how="inner").reset_index(
            drop=True
        )
        if len(merged) < 4:
            continue
        rows.append(
            {"sample_id": sample_id, "frame": merged, "meta": sample.meta, "root": sample.root}
        )
    return rows


def _selection_lpips(frame: pd.DataFrame, score_column: str) -> float:
    return float(frame.loc[frame[score_column].idxmax(), "lpips"])


def e2_ablation(
    cfg: Cfg, experiment: str, split: str, terms: list[str] | None = None
) -> pd.DataFrame:
    """Per-term single selections and leave-one-out combinations (mean LPIPS + CI)."""
    rows = _load_rows(cfg, experiment, split)
    if not rows:
        raise ValueError(f"no scored+measured samples under {experiment}/{split}")
    terms = terms or [name for name in DEFAULT_TERMS if f"{name}_z" in rows[0]["frame"].columns]

    configs: dict[str, list[float]] = {f"single_{name}": [] for name in terms}
    configs |= {f"loo_{name}": [] for name in terms}
    configs["full_combined_A"] = []
    configs["oracle"] = []
    configs["random_expected"] = []

    for row in rows:
        frame = row["frame"]
        lpips = frame["lpips"].to_numpy(dtype=np.float64)
        configs["oracle"].append(float(lpips.min()))
        configs["random_expected"].append(float(lpips.mean()))
        configs["full_combined_A"].append(_selection_lpips(frame, "combined_A"))
        for name in terms:
            configs[f"single_{name}"].append(_selection_lpips(frame, f"{name}_z"))
            others = [n for n in terms if n != name]
            combined = np.zeros(len(frame))
            for other in others:
                combined += frame[f"{other}_z"].to_numpy(dtype=np.float64)
            configs[f"loo_{name}"].append(float(lpips[int(np.argmax(combined))]))

    records = []
    for name, values in configs.items():
        arr = np.asarray(values, dtype=np.float64)
        ci = bootstrap_ci(arr, n_boot=cfg.eval.pairwise_bootstrap)
        records.append(
            {
                "configuration": name,
                "n_images": len(arr),
                "mean_lpips": round(ci["mean"], 4),
                "ci_lo": round(ci["lo"], 4),
                "ci_hi": round(ci["hi"], 4),
                "median_lpips": round(float(np.median(arr)), 4),
            }
        )
    return pd.DataFrame(records).sort_values("mean_lpips").reset_index(drop=True)


def e3_scaling(
    cfg: Cfg,
    experiment: str,
    split: str,
    ns: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    n_perm: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean LPIPS of mode-A selection as N grows (random subset permutations)."""
    rows = _load_rows(cfg, experiment, split)
    if not rows:
        raise ValueError(f"no scored+measured samples under {experiment}/{split}")
    rng = np.random.default_rng(seed)

    records = []
    for n in ns:
        selected, oracle = [], []
        for row in rows:
            frame = row["frame"]
            values = frame["combined_A"].to_numpy(dtype=np.float64)
            lpips = frame["lpips"].to_numpy(dtype=np.float64)
            if len(frame) < n:
                continue
            for _ in range(n_perm):
                subset = rng.choice(len(frame), size=n, replace=False)
                pick = int(subset[int(np.argmax(values[subset]))])
                selected.append(float(lpips[pick]))
                oracle.append(float(lpips[subset].min()))
        ci = bootstrap_ci(selected, n_boot=cfg.eval.pairwise_bootstrap)
        records.append(
            {
                "n_candidates": n,
                "mean_lpips_selected": round(ci["mean"], 4),
                "ci_lo": round(ci["lo"], 4),
                "ci_hi": round(ci["hi"], 4),
                "mean_lpips_oracle": round(float(np.mean(oracle)), 4),
                "n_draws": len(selected),
            }
        )
    return pd.DataFrame(records)


def e4_calibration(cfg: Cfg, experiment: str, split: str, top_frac: float = 0.1) -> dict[str, Any]:
    """Uncertainty calibration: pixel Spearman + AUROC and image-level Spearman."""
    from scipy import stats as sps
    from sklearn.metrics import roc_auc_score

    from counterpart.segment.threshold import delta_e76, rgb_to_lab

    runs_dir = Path(cfg.project.paths.runs_dir)
    pixel_spearman: list[float] = []
    pixel_auroc: list[float] = []
    image_rows: list[tuple[float, float]] = []

    for sample_id in cio.iter_sample_ids(runs_dir, experiment, split):
        sample = cio.load_sample(runs_dir, experiment, split, sample_id)
        if (
            not sample.uncertainty_path.exists()
            or not sample.selection_path.exists()
            or not (sample.root / "metrics.parquet").exists()
        ):
            continue
        selection = cio.load_json(sample.selection_path)
        best_idx = selection.get("ours_eq")
        if best_idx is None:
            continue
        uncertainty = np.load(sample.uncertainty_path)
        damage_mask = cio.read_mask(sample.damage_mask_path)
        if int(damage_mask.sum()) < 64:
            continue

        gt = cio.read_image(sample.original_path)
        best = cio.read_image(sample.candidate_path(int(best_idx)))
        error = delta_e76(rgb_to_lab(gt), rgb_to_lab(best))

        u = uncertainty[damage_mask].astype(np.float64)
        e = error[damage_mask].astype(np.float64)
        if np.allclose(u, 0.0) or np.allclose(e, 0.0):
            continue
        pixel_spearman.append(float(sps.spearmanr(u, e).statistic))
        threshold = float(np.percentile(e, 100 * (1 - top_frac)))
        labels = (e >= threshold).astype(int)
        if labels.min() != labels.max():
            pixel_auroc.append(float(roc_auc_score(labels, u)))

        metrics = pd.read_parquet(sample.root / "metrics.parquet")
        best_lpips = float(metrics.loc[metrics["idx"] == int(best_idx), "lpips"].iloc[0])
        image_rows.append((float(uncertainty[damage_mask].mean()), best_lpips))

    summary: dict[str, Any] = {
        "split": split,
        "n_images": len(pixel_spearman),
        "mean_pixel_spearman": round(float(np.mean(pixel_spearman)), 4) if pixel_spearman else None,
        "mean_pixel_auroc_top10": round(float(np.mean(pixel_auroc)), 4) if pixel_auroc else None,
    }
    if image_rows:
        means, lpips = zip(*image_rows, strict=True)
        summary["image_level_spearman"] = round(
            float(sps.spearmanr(np.asarray(means), np.asarray(lpips)).statistic), 4
        )
    return summary


def e5_breakdowns(cfg: Cfg, experiment: str, split: str) -> dict[str, pd.DataFrame]:
    """Selector performance grouped by damage type, area bucket and product type."""
    results_dir = Path(cfg.project.paths.runs_dir) / experiment / "results"
    per_image = pd.read_parquet(results_dir / f"per_image_{split}.parquet")

    selectors = [
        ("ours_eq", "lpips_sel_ours_eq"),
        ("ours_learned", "lpips_sel_ours_learned"),
        ("clip", "lpips_sel_clip"),
        ("dino", "lpips_sel_dino"),
        ("first", "lpips_sel_first"),
    ]
    available = [(name, col) for name, col in selectors if col in per_image.columns]
    per_image = per_image.copy()
    per_image["area_bucket"] = pd.cut(
        per_image["area_frac"], bins=[0.0, 0.15, 0.22, 1.0], labels=["10-15%", "15-22%", "22-30%"]
    )

    out: dict[str, pd.DataFrame] = {}
    for group in ("damage_type", "area_bucket", "product_type"):
        rows = []
        for value, chunk in per_image.groupby(group, observed=True):
            row: dict[str, Any] = {group: str(value), "n": len(chunk)}
            for name, column in available:
                values = chunk[column].dropna()
                if len(values):
                    row[f"mean_lpips_{name}"] = round(float(values.mean()), 4)
            if "lpips_oracle" in chunk.columns:
                row["mean_lpips_oracle"] = round(float(chunk["lpips_oracle"].mean()), 4)
            if "lpips_random" in chunk.columns:
                row["mean_lpips_random"] = round(float(chunk["lpips_random"].mean()), 4)
            rows.append(row)
        out[group] = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return out


def run_experiments(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    which: tuple[str, ...] = ("E2", "E3", "E4", "E5"),
) -> dict[str, Any]:
    """Run the requested cache-only experiments and write their tables."""
    results_dir = Path(cfg.project.paths.runs_dir) / experiment / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"split": split, "experiment": experiment}

    if "E2" in which:
        table = e2_ablation(cfg, experiment, split)
        table.to_csv(results_dir / f"e2_ablation_{split}.csv", index=False)
        summary["e2"] = table.head(6).to_dict(orient="records")
    if "E3" in which:
        table = e3_scaling(cfg, experiment, split)
        table.to_csv(results_dir / f"e3_scaling_{split}.csv", index=False)
        summary["e3"] = table.to_dict(orient="records")
    if "E4" in which:
        summary["e4"] = e4_calibration(cfg, experiment, split)
    if "E5" in which:
        breakdowns = e5_breakdowns(cfg, experiment, split)
        for name, table in breakdowns.items():
            table.to_csv(results_dir / f"e5_breakdown_{name}_{split}.csv", index=False)
        summary["e5_tables"] = list(breakdowns)
    return summary
