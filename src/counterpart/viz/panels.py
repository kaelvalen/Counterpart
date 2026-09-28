"""Visualization helpers (SPEC.md §5.5). Expanded in later phases."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def contact_sheet(
    images: list[np.ndarray],
    titles: list[str],
    path: Path,
    cols: int = 8,
    dpi: int = 110,
) -> Path:
    """Save a grid of images with titles (used for QA panels and candidate sheets)."""
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length")
    if not images:
        raise ValueError("no images to plot")

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, title in zip(axes, images, titles, strict=False):
        ax.imshow(img)
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
