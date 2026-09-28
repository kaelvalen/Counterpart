"""Configuration: pydantic v2 models, YAML loading, profile merging.

Usage::

    cfg = load_config(
        Path("configs/default.yaml"),
        profile=Path("configs/profiles/laptop_8gb.yaml"),
    )

The profile is deep-merged on top of the base config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    """Base model: unknown keys are a hard error (catch config typos early)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- project


class PathsCfg(Section):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")


class ProjectCfg(Section):
    name: str = "counterpart"
    seed: int = 0
    paths: PathsCfg = Field(default_factory=PathsCfg)


# ------------------------------------------------------------------------- generation


class GeneratorCfg(Section):
    """Frozen diffusion inpainting model settings (SPEC.md §3.4, §5.3)."""

    id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    optional_id: str | None = (
        None  # e.g. stabilityai/stable-diffusion-2-inpainting if access opens up
    )
    dtype: Literal["float16", "float32"] = "float16"
    resolution: int = 512
    n_candidates: int = 32
    num_inference_steps: int = 25
    guidance_scales: list[float] = Field(default_factory=lambda: [4.0, 7.5])
    prompts: list[str] = Field(
        default_factory=lambda: [
            "",
            "a complete intact object, product photo, white background",
            "a complete intact {category}, product photo",
        ]
    )
    negative_prompt: str = "broken, cracked, chipped, damaged, missing piece, hole, shattered"
    batch_size: int = 1
    attention_slicing: bool = True
    seam_feather_px: int = 3
    seed: int = 0  # base seed; per-candidate seed = hash(sample_id) + idx + seed


# ---------------------------------------------------------------------------- damage


class DamageCfg(Section):
    """Synthetic damage generator settings (SPEC.md §6.2)."""

    type_probs: dict[str, float] = Field(
        default_factory=lambda: {"chip": 0.70, "hole": 0.15, "crack": 0.15}
    )
    area_frac_min: float = 0.10
    area_frac_max: float = 0.30
    gen_mask_dilate_px: int = 6
    fractal_levels: int = 4
    fractal_roughness: float = 0.35
    crack_width_px: tuple[int, int] = (3, 8)
    fracture_rim: bool = False  # optional realism (robustness experiment)


# ------------------------------------------------------------------------------ root


class Cfg(Section):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    damage: DamageCfg = Field(default_factory=DamageCfg)


# -------------------------------------------------------------------------- loading


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, profile: str | Path | None = None) -> Cfg:
    """Load YAML config, optionally deep-merging a profile file on top of it."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    if profile is not None:
        profile_path = Path(profile)
        prof = yaml.safe_load(profile_path.read_text()) or {}
        if not isinstance(prof, dict):
            raise ValueError(f"profile root must be a mapping: {profile_path}")
        raw = _deep_merge(raw, prof)
    return Cfg.model_validate(raw)
