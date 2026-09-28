"""Combining and scoring pipeline tests (SPEC.md §5.4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.score.combine import combine_terms, pairwise_logistic_fit, z_normalize
from counterpart.score.run import score_sample


def test_z_normalize_masks_non_applicable() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    applicable = np.array([True, True, False, False])
    z = z_normalize(values, applicable)
    assert z[2] == 0.0 and z[3] == 0.0
    assert z[0] == pytest.approx(-1.0)
    assert z[1] == pytest.approx(1.0)


def test_z_normalize_constant_is_zero() -> None:
    values = np.full(4, 7.0)
    assert np.allclose(z_normalize(values, np.ones(4, dtype=bool)), 0.0)


def test_combine_renormalises_weights_for_inapplicable_terms() -> None:
    n = 4
    raw = {
        "T1": np.array([0.0, 1.0, 2.0, 3.0]),
        "T2": np.full(n, np.nan),  # not applicable at all
        "T6": np.array([3.0, 2.0, 1.0, 0.0]),
    }
    applicable = {
        "T1": np.ones(n, dtype=bool),
        "T2": np.zeros(n, dtype=bool),
        "T6": np.ones(n, dtype=bool),
    }
    combined, z = combine_terms(raw, applicable, weights=None)
    assert np.isfinite(combined).all(), "NaN term must not poison the combination"
    assert np.allclose(z["T2"], 0.0)
    assert np.allclose(combined, 0.0, atol=1e-9)


def test_pairwise_logistic_prefers_predictive_feature() -> None:
    rng = np.random.default_rng(0)
    n = 60
    predictive = rng.normal(size=n)
    noise = rng.normal(size=n)
    lpips = -predictive + 0.1 * rng.normal(size=n)  # predictive feature explains quality
    features = np.stack([predictive, noise], axis=1)
    weights = pairwise_logistic_fit(features, lpips, l2=0.01, iters=300)
    assert weights[0] > weights[1]


def _write_candidate_sample(root: Path) -> dict[str, np.ndarray]:
    """Tiny sample dir with 4 candidates; returns the candidate images."""
    import cv2

    from counterpart.score.base import build_score_sample  # noqa: F401  (import sanity)
    from counterpart.segment.threshold import object_mask

    size = 192
    cfg = Cfg()
    intact = np.full((size, size, 3), 248, dtype=np.uint8)
    cv2.ellipse(intact, (size // 2 - 6, size // 2), (48, 60), 0, 0, 360, (90, 70, 150), -1)
    obj = object_mask(intact, cfg.segment)
    damage = np.zeros((size, size), dtype=bool)
    damage[80:130, 128:170] = True
    damage &= obj
    damaged = intact.copy()
    damaged[damage] = 248
    rng = np.random.default_rng(0)
    noise = intact.copy()
    noise[damage] = rng.integers(0, 256, size=(int(damage.sum()), 3), dtype=np.uint8)

    root.mkdir(parents=True, exist_ok=True)
    cio.write_image(root / "original.png", intact)
    cio.write_image(root / "damaged.png", damaged)
    cio.write_mask(root / "damage_mask.png", damage)
    cio.write_mask(root / "gen_mask.png", damage)
    cio.write_mask(root / "object_mask.png", obj)
    cio.save_json(root / "meta.json", {"sample_id": "s0", "split": "test"})

    candidates = {"gt": intact, "white": damaged, "noise": noise}
    records = []
    for idx, image in enumerate(candidates.values()):
        cio.write_image(root / "candidates" / f"cand_{idx:03d}.png", image)
        records.append(
            {
                "idx": idx,
                "path": str(root / "candidates" / f"cand_{idx:03d}.png"),
                "seed": idx,
                "prompt": "",
                "guidance_scale": 7.5,
                "num_inference_steps": 25,
                "generator_id": "test",
            }
        )
    cio.append_jsonl(root / "candidates.jsonl", records)
    return candidates


def test_gt_injection_ranks_ground_truth_near_top(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    sample_dir = runs_dir / "main" / "test" / "s0"
    _write_candidate_sample(sample_dir)

    cfg = Cfg()
    cfg.project.paths.runs_dir = runs_dir  # type: ignore[assignment]
    score_sample(cfg, "main", "test", "s0", terms=["T1", "T6"])

    from counterpart.eval.gt_injection import run_gt_injection

    summary = run_gt_injection(cfg, "test", experiment="main")
    assert summary["n_images"] == 1
    assert summary["mean_combined_percentile"] <= 0.5, (
        "the ground truth must not rank at the bottom of its own candidate pool"
    )


def test_score_sample_ranks_ground_truth_first(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    sample_dir = runs_dir / "main" / "test" / "s0"
    _write_candidate_sample(sample_dir)

    cfg = Cfg()
    cfg.project.paths.runs_dir = runs_dir  # type: ignore[assignment]
    frame = score_sample(cfg, "main", "test", "s0", terms=["T1", "T6"])
    assert {"T1_raw", "T6_raw", "combined_A"} <= set(frame.columns)
    best = int(frame["combined_A"].idxmax())
    assert best == 0, "ground-truth candidate must rank first"
    assert frame.loc[0, "T1_applicable"]
    cached = score_sample(cfg, "main", "test", "s0", terms=["T1", "T6"])
    assert cached.equals(frame), "second run must hit the cache"
    # diagnostics are stored as JSON strings
    assert json.loads(frame.loc[0, "T1_diag"])["n_contacts"] >= 0
