"""Combining score terms into one ranking (SPEC.md §5.4, "Birleştirme").

* Each term is z-normalised **within the image** over its applicable candidates
  (selection is an intra-image ranking problem).
* Terms with ``applicable=False`` for an image (or with zero variance) get weight 0;
  the remaining weights are renormalised.
* Mode A (main result): equal weights — no training, no leakage.
* Mode B (secondary): weights fitted on the *train* split by pairwise logistic
  regression against GT LPIPS (RankNet-lite, L2). This module is the only
  ``score/`` code allowed to read ground truth, and only for the train split fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_TERMS = ["T1", "T6", "T4", "T5", "T2", "T3"]


def z_normalize(values: np.ndarray, applicable: np.ndarray) -> np.ndarray:
    """Z-normalise ``values`` over the applicable entries; others become 0."""
    out = np.zeros_like(values, dtype=np.float64)
    mask = applicable & np.isfinite(values)
    if int(mask.sum()) < 2:
        return out
    mu = float(values[mask].mean())
    sd = float(values[mask].std())
    if sd < 1e-9:
        return out
    out[mask] = (values[mask] - mu) / sd
    return out


def combine_terms(
    raw: dict[str, np.ndarray],
    applicable: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (combined score, per-term z-values).

    ``weights=None`` → mode A (equal weights over terms with at least one applicable,
    finite candidate).
    """
    names = list(raw)
    n = len(raw[names[0]])
    z_values = {
        name: z_normalize(np.asarray(raw[name], dtype=np.float64), applicable[name])
        for name in names
    }

    combined = np.zeros(n, dtype=np.float64)
    total_weight = 0.0
    for name in names:
        active = bool(
            applicable[name].any() and np.isfinite(np.asarray(raw[name], dtype=np.float64)).any()
        )
        if not active:
            continue
        weight = 1.0 if weights is None else float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        combined += weight * z_values[name]
        total_weight += weight
    if total_weight > 0:
        combined /= total_weight
    return combined, z_values


def pairwise_logistic_fit(
    features: np.ndarray,
    lpips: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    l2: float = 1.0,
    iters: int = 400,
    lr: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """RankNet-lite: pairwise logistic regression on (better, worse) candidate pairs.

    ``features`` is (N, T) z-normalised term values, ``lpips`` (N,) with lower = better.
    ``groups`` assigns each row to an image; pairs are only formed within a group.
    Returns non-negative weights (T,).
    """
    features = np.asarray(features, dtype=np.float64)
    lpips = np.asarray(lpips, dtype=np.float64)
    n, t = features.shape
    if groups is None:
        groups = np.zeros(n, dtype=int)
    groups = np.asarray(groups)

    better_idx: list[np.ndarray] = []
    worse_idx: list[np.ndarray] = []
    for group in np.unique(groups):
        members = np.nonzero(groups == group)[0]
        if len(members) < 2:
            continue
        i, j = np.triu_indices(len(members), k=1)
        first, second = members[i], members[j]
        swap = lpips[second] < lpips[first]
        better_idx.append(np.where(swap, second, first))
        worse_idx.append(np.where(swap, first, second))
    if not better_idx:
        return np.ones(t) / t
    better = np.concatenate(better_idx)
    worse = np.concatenate(worse_idx)

    x = features[better] - features[worse]  # (P, T); positive margin = first should win
    weights = np.ones(t) / t
    for _ in range(iters):
        margins = x @ weights
        # logistic loss gradient: d/dw log(1 + exp(-m)) = -sigmoid(-m) * x
        sig = 1.0 / (1.0 + np.exp(np.clip(margins, -30, 30)))
        grad = -(sig[:, None] * x).mean(axis=0) + l2 * weights
        weights -= lr * grad
        weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    return weights / total if total > 0 else np.ones(t) / t


def train_z_features(term_matrix: np.ndarray, applicability: np.ndarray) -> np.ndarray:
    """Z-normalise a per-candidate term matrix (N images x C candidates x T terms) per image."""
    n_images, n_candidates, t = term_matrix.shape
    out = np.zeros_like(term_matrix, dtype=np.float64)
    for i in range(n_images):
        for term in range(t):
            out[i, :, term] = z_normalize(term_matrix[i, :, term], applicability[i, :, term])
    return out


def save_weights(path: Path, weights: dict[str, float], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"weights": weights, "meta": meta}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def load_weights(path: Path) -> tuple[dict[str, float], dict]:
    payload = json.loads(Path(path).read_text())
    return payload["weights"], payload.get("meta", {})
