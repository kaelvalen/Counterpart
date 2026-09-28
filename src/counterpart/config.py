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


# ------------------------------------------------------------------------------ data


class DataCfg(Section):
    """ABO subset selection and split sizes (SPEC.md §6.1, §6.3)."""

    data_dir: Path = Path("data/abo")
    product_types: list[str] = Field(
        default_factory=lambda: [
            "VASE",
            "CUP",
            "MUG",
            "BOWL",
            "LAMP",
            "TOY",
            "KITCHEN",
            "HOME_FURNITURE_AND_DECOR",
        ]
    )
    exclude_keywords: list[str] = Field(
        default_factory=lambda: [
            "shirt",
            "t-shirt",
            "dress",
            "sock",
            "pants",
            "hoodie",
            "jacket",
            "blanket",
            "towel",
            "curtain",
            "pillow",
            "sheet",
            "duvet",
        ]
    )
    min_side: int = 400
    object_area_frac: tuple[float, float] = (0.15, 0.70)
    split_sizes: dict[str, int] = Field(
        default_factory=lambda: {"gonogo": 100, "train": 300, "val": 100, "test": 300}
    )
    oversample_factor: float = 2.0  # candidate products per needed sample (rejections)
    seed: int = 7


# --------------------------------------------------------------------------- segment


class SegmentCfg(Section):
    """Object silhouette from background thresholding (SPEC.md §5.1)."""

    border_px: int = 8
    background_delta_e: float = 10.0  # pixel counts as object when ΔE76 to bg exceeds this
    whiteness_delta_e: float = 6.0  # border pixels within this of bg colour = "white background"
    whiteness_frac: float = 0.95
    close_kernel: int = 5
    min_area_frac: float = 0.05
    max_area_frac: float = 0.90
    crop_margin: float = 0.15
    size: int = 512


# ------------------------------------------------------------------------------- eval


class EvalCfg(Section):
    """Ground-truth metrics and selection-quality settings (SPEC.md §8)."""

    lpips_net: str = "alex"
    lpips_crop_margin: int = 16
    lpips_min_crop: int = 128  # alexnet cannot handle very small crops
    boundary_ring_px: int = 8
    delta_e: str = "76"  # "76" | "2000"
    pairwise_bootstrap: int = 1000


# ------------------------------------------------------------------------------ root


class Cfg(Section):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    damage: DamageCfg = Field(default_factory=DamageCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    segment: SegmentCfg = Field(default_factory=SegmentCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)


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
