"""Mode discovery (SPEC.md §5.5).

Each candidate's damage-region crop (everything outside M in neutral grey) is embedded
with DINOv2; agglomerative clustering with cosine distance and average linkage groups
the candidates into "modes". Every cluster's representative is its highest-scoring
(combined_A) candidate, and the cluster size is reported as a probability mass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.eval.metrics import crop_box_for_mask


def _crop_for_embedding(
    candidate: np.ndarray, damage_mask: np.ndarray, margin: int, gray: int
) -> np.ndarray:
    """Damage-region crop with everything outside M set to neutral grey."""
    masked = candidate.copy()
    masked[~damage_mask] = gray
    y0, y1, x0, x1 = crop_box_for_mask(damage_mask, margin=margin, min_side=96)
    return masked[y0:y1, x0:x1]


def compute_modes_sample(
    cfg: Cfg, experiment: str, split: str, sample_id: str, *, overwrite: bool = False
) -> dict[str, Any]:
    """Cluster the candidates of one sample into modes; writes ``modes.json``."""
    from sklearn.cluster import AgglomerativeClustering
    from transformers import AutoImageProcessor, AutoModel

    runs_dir = Path(cfg.project.paths.runs_dir)
    sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
    out_path = sample.modes_path
    if out_path.exists() and not overwrite:
        return cio.load_json(out_path)

    scores = pd.read_parquet(sample.scores_path)
    damage_mask = cio.read_mask(sample.damage_mask_path)

    order = scores.sort_values("combined_A", ascending=False)["idx"].tolist()
    crops, indices = [], []
    for idx in order:
        candidate = cio.read_image(sample.candidate_path(int(idx)))
        crops.append(
            _crop_for_embedding(
                candidate, damage_mask, cfg.select.mode_crop_margin, cfg.baselines.neutral_gray
            )
        )
        indices.append(int(idx))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(cfg.select.mode_embedding)
    model = AutoModel.from_pretrained(cfg.select.mode_embedding).to(device).eval()
    try:
        embeddings = []
        for crop in crops:
            inputs = processor(images=Image.fromarray(crop), return_tensors="pt").to(device)
            with torch.inference_mode():
                cls = model(**inputs).last_hidden_state[:, 0]
            embeddings.append(cls.squeeze(0).float().cpu().numpy())
        embeddings = np.stack(embeddings)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-9)
    distance = 1.0 - normalized @ normalized.T
    np.fill_diagonal(distance, 0.0)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=cfg.select.mode_distance_threshold,
    )
    labels = clustering.fit_predict(distance)

    modes: list[dict[str, Any]] = []
    for label in sorted(set(int(x) for x in labels)):
        members = [i for i, assigned in enumerate(labels) if int(assigned) == label]
        modes.append(
            {
                "mode": int(label),
                "size": len(members),
                "probability_mass": round(len(members) / len(indices), 4),
                "representative_idx": indices[members[0]],  # highest combined_A in cluster
                "indices": sorted(indices[i] for i in members),
            }
        )
    modes.sort(key=lambda m: (-m["size"], m["mode"]))

    payload = {
        "sample_id": sample_id,
        "threshold": cfg.select.mode_distance_threshold,
        "n_modes": len(modes),
        "modes": modes,
    }
    cio.save_json(out_path, payload)
    return payload


def run_modes(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    limit: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute modes for all samples of a split (serial: one GPU model at a time)."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no samples under {runs_dir / experiment / split}")

    mode_counts = []
    for i, sample_id in enumerate(sample_ids, start=1):
        payload = compute_modes_sample(cfg, experiment, split, sample_id, overwrite=overwrite)
        mode_counts.append(payload["n_modes"])
        if verbose:
            print(f"[{i}/{len(sample_ids)}] {sample_id}: {payload['n_modes']} modes")
    return {
        "split": split,
        "experiment": experiment,
        "samples": len(sample_ids),
        "mean_modes": round(float(np.mean(mode_counts)), 3) if mode_counts else 0.0,
    }
