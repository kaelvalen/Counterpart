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


# ------------------------------------------------------------------------------ score


class ScoreCfg(Section):
    """Classical consistency score terms (SPEC.md §5.4). All thresholds live here."""

    ring_width_px: int = 8
    delta_e: str = "76"  # "76" | "2000"

    # T1 — boundary continuity
    t1_sample_step_px: int = 4
    t1_seam_depth_px: int = 4
    t1_seam_scale: float = 20.0  # ΔE that counts as fully inconsistent
    t1_gradient_scale: float = 40.0  # grey-level gradient magnitude difference
    t1_seam_weight: float = 1 / 3
    t1_gradient_weight: float = 1 / 3
    t1_dangling_weight: float = 1 / 3
    t1_canny_low: int = 50
    t1_canny_high: int = 150
    t1_contact_radius_px: int = 2
    t1_contact_max_alignment: float = 0.77  # |g·n| below this = edge arrives ⊥ to ∂M
    t1_min_contacts: int = 3

    # T2 — mirror symmetry
    t2_angle_step_deg: float = 2.0
    t2_confidence_threshold: float = 0.6
    t2_min_overlap_px: int = 400
    t2_color_scale: float = 15.0
    t2_offset_steps: int = 3
    t2_offset_frac: float = 0.02
    t2_refine_iters: int = 2

    # T3 — rotational symmetry
    t3_k_min: int = 2
    t3_k_max: int = 24
    t3_confidence_threshold: float = 0.6
    t3_radial_bins: int = 24
    t3_angular_bins: int = 360
    t3_color_scale: float = 15.0
    t3_continuous_tol: float = 0.05
    t3_hough_min_radius_frac: float = 0.15
    t3_hough_max_radius_frac: float = 0.75

    # T4 — texture statistics
    t4_patch_px: int = 32
    t4_min_mask_pixels: int = 400
    t4_min_patches: int = 6
    t4_lbp_weight: float = 1 / 3
    t4_gabor_weight: float = 1 / 3
    t4_color_weight: float = 1 / 3
    t4_lbp_scale: float = 0.2
    t4_gabor_scale: float = 0.5
    t4_color_scale: float = 0.2
    t4_superpixel_regions: int = 24

    # T5 — frequency profile
    t5_patch_px: int = 32
    t5_min_patches: int = 6
    t5_high_freq_cutoff: float = 0.35
    t5_slope_scale: float = 0.5
    t5_hf_scale: float = 0.2

    # T6 — contour continuity
    t6_fit_radius_px: int = 14
    t6_contact_px: float = 2.0
    t6_contact_min_separation_px: int = 6
    t6_tangent_weight: float = 1.0
    t6_curvature_lambda: float = 0.5
    t6_curvature_scale: float = 1.0
    t6_min_segment_px: int = 8


# -------------------------------------------------------------------------- baselines


class BaselinesCfg(Section):
    """Baseline inpainting and learned selectors (SPEC.md §7)."""

    classical_methods: list[str] = Field(default_factory=lambda: ["telea", "ns"])
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_prompt: str = "a photo of an intact {category}"
    dino_model: str = "facebook/dinov2-base"
    dino_cls_weight: float = 0.5
    dino_nn_weight: float = 0.5
    neutral_gray: int = 128


# ---------------------------------------------------------------------------- select


class SelectCfg(Section):
    """Selection, mode discovery and uncertainty (SPEC.md §5.5)."""

    mode_distance_threshold: float = 0.35
    mode_embedding: str = "facebook/dinov2-base"
    mode_crop_margin: int = 32
    uncertainty_top_k_variant: int = 8
    random_seed: int = 0
    panel_modes: int = 3


# ------------------------------------------------------------------------------ root


class Cfg(Section):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    damage: DamageCfg = Field(default_factory=DamageCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    segment: SegmentCfg = Field(default_factory=SegmentCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    score: ScoreCfg = Field(default_factory=ScoreCfg)
    baselines: BaselinesCfg = Field(default_factory=BaselinesCfg)
    select: SelectCfg = Field(default_factory=SelectCfg)


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
