"""Report figures for the experiments (E0 first; extended in later phases)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from counterpart import io as cio  # noqa: E402
from counterpart.eval.metrics import crop_box_for_mask  # noqa: E402
from counterpart.viz import panels  # noqa: E402
from counterpart.viz.panels import contact_sheet  # noqa: E402

Box = tuple[int, int, int, int]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _crop_zoom(image: np.ndarray, box: Box, scale: int = 2) -> np.ndarray:
    """Crop the damage region and upscale (nearest) so small defects stay visible."""
    y0, y1, x0, x1 = box
    return cv2.resize(
        image[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )


def e0_figures(
    per_image: pd.DataFrame,
    all_candidate_lpips: list[np.ndarray],
    results_dir: Path,
    n_examples: int = 10,
) -> list[Path]:
    """E0 output figures: LPIPS histogram, per-image spread, selector comparison, examples."""
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    flat = np.concatenate(all_candidate_lpips) if all_candidate_lpips else np.array([])

    # 1) candidate LPIPS distribution
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.hist(flat, bins=40, density=True, alpha=0.75, color="#4878a8")
    ax.axvline(
        float(per_image["lpips_random"].mean()), color="#c44", ls="--", label="random (mean)"
    )
    ax.axvline(
        float(per_image["lpips_oracle"].mean()), color="#2a2", ls="--", label="oracle (mean)"
    )
    ax.set_xlabel("candidate LPIPS")
    ax.set_ylabel("density")
    ax.set_title("E0: candidate LPIPS distribution (all images, all candidates)")
    ax.legend()
    produced.append(_save(fig, figures_dir / "e0_lpips_hist.png"))

    # 2) per-image spread
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.hist(per_image["spread"], bins=25, alpha=0.75, color="#8a7248")
    ax.axvline(float(per_image["spread"].mean()), color="#c44", ls="--", label="mean spread")
    ax.set_xlabel("std of candidate LPIPS within an image")
    ax.set_ylabel("#images")
    ax.set_title("E0: within-image quality spread")
    ax.legend()
    produced.append(_save(fig, figures_dir / "e0_spread_hist.png"))

    # 3) selector comparison
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    keys = ["lpips_oracle", "lpips_first", "lpips_random", "lpips_worst"]
    labels = ["oracle", "first", "random (expected)", "worst"]
    values = [float(per_image[k].mean()) for k in keys]
    bars = ax.bar(labels, values, color=["#2a8f2a", "#4878a8", "#888888", "#a85048"])
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_ylabel("mean LPIPS")
    ax.set_title("E0: selector comparison (lower is better)")
    produced.append(_save(fig, figures_dir / "e0_selectors.png"))

    # 4) example panels (damage-region crops), biased to images with the largest spread
    order = per_image.sort_values("spread", ascending=False).head(n_examples)
    strips, titles = [], []
    for row in order.itertuples():
        root = results_dir.parent / str(row.split) / str(row.sample_id)
        if not root.exists():
            continue
        damaged = cio.read_image(root / "damaged.png")
        gt = cio.read_image(root / "original.png")
        mask = cio.read_mask(root / "damage_mask.png")
        box = crop_box_for_mask(mask, margin=32, min_side=96)
        cand_imgs = [
            cio.read_image(root / "candidates" / f"cand_{int(idx):03d}.png")
            for idx in (row.oracle_idx, row.worst_idx)
        ]
        strip = [_crop_zoom(im, box) for im in [damaged, gt, *cand_imgs]]
        strips.append(panels.side_by_side(strip))
        titles.append(
            f"{row.sample_id} | oracle #{int(row.oracle_idx)} vs worst #{int(row.worst_idx)}"
        )
    if strips:
        produced.append(contact_sheet(strips, titles, figures_dir / "e0_examples.png", cols=2))

    return produced
