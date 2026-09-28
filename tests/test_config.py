"""Config loading: defaults, profile deep-merge, unknown-key rejection."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from counterpart.config import load_config


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_defaults_load(tmp_path: Path) -> None:
    base = _write(tmp_path, "base.yaml", "generator:\n  num_inference_steps: 25\n")
    cfg = load_config(base)
    assert cfg.generator.num_inference_steps == 25
    assert cfg.generator.id == "stable-diffusion-v1-5/stable-diffusion-inpainting"
    assert cfg.damage.gen_mask_dilate_px == 6


def test_profile_deep_merge(tmp_path: Path) -> None:
    base = _write(
        tmp_path,
        "base.yaml",
        "generator:\n  num_inference_steps: 25\n  guidance_scales: [4.0, 7.5]\n",
    )
    profile = _write(tmp_path, "profile.yaml", "generator:\n  guidance_scales: [3.0]\n")
    cfg = load_config(base, profile=profile)
    assert cfg.generator.guidance_scales == [3.0]
    assert cfg.generator.num_inference_steps == 25, "non-overridden keys must survive merging"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = _write(tmp_path, "bad.yaml", "generator:\n  num_steps: 25\n")
    with pytest.raises(ValidationError):
        load_config(bad)
