"""Web-app path integration test: generate -> score -> select on one image+mask."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from counterpart.config import Cfg
from counterpart.webapp import reconstruct

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU pipeline")


def test_reconstruct_returns_panel_ready_arrays(tmp_path: Path) -> None:
    cfg = Cfg()
    cfg.project.paths.runs_dir = tmp_path / "runs"  # type: ignore[assignment]
    cfg.generator.n_candidates = 2
    cfg.generator.num_inference_steps = 6

    size = 256
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    cv2.ellipse(image, (size // 2, size // 2), (60, 80), 0, 0, 360, (80, 90, 160), -1)
    mask = np.zeros((size, size), dtype=bool)
    mask[100:150, 140:190] = True
    mask &= image.sum(axis=2) < 700  # only inside the object

    result = reconstruct(cfg, image, mask, n=2)
    assert result["best"].shape == (512, 512, 3)
    assert result["uncertainty"].shape == (512, 512, 3)
    assert result["candidates"].shape[0] == 512
    assert float(result["mean_uncertainty"]) >= 0.0
