"""Frozen SD inpainting wrapper (SPEC.md §5.3).

* Loads the configured diffusers inpainting pipeline once; callers generate candidates
  in a loop and release the model when the stage ends (stages never share the GPU).
* Post-process: everything outside ``gen_mask`` is restored bit-exactly from the
  damaged input; a ``seam_feather_px``-wide band centred on the seam is blended so
  the splice is not a hard edge (SPEC.md §5.3).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

from counterpart.config import GeneratorCfg
from counterpart.types import CandidateSpec


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate a bool mask by ``px`` pixels (elliptical structuring element)."""
    if px <= 0:
        return mask.copy()
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def blend_seam(
    original: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    feather_px: int,
) -> np.ndarray:
    """Splice ``candidate`` into ``original`` inside ``mask`` with a feather band.

    Outside ``dilate(mask, feather_px)`` the result is exactly ``original``.
    Inside ``erode(mask, feather_px)`` it is exactly ``candidate``. In between a
    linear ramp on the signed distance to the boundary blends the two.
    """
    if feather_px <= 0:
        return np.where(mask[..., None], candidate, original)

    m = mask.astype(np.uint8)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    signed = dist_out - dist_in  # >0 outside the mask, <0 inside
    alpha = np.clip(0.5 - signed / (2.0 * feather_px), 0.0, 1.0).astype(np.float32)
    alpha = alpha[..., None]
    out = original.astype(np.float32) * (1.0 - alpha) + candidate.astype(np.float32) * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class InpaintGenerator:
    """Thin wrapper around a diffusers inpainting pipeline."""

    def __init__(self, pipe: AutoPipelineForInpainting, cfg: GeneratorCfg, device: str = "cuda"):
        self.pipe = pipe
        self.cfg = cfg
        self.device = device

    # ------------------------------------------------------------------------ load

    @classmethod
    def load(cls, cfg: GeneratorCfg, device: str = "cuda") -> InpaintGenerator:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; cannot load the inpainting pipeline")
        dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                cfg.id, dtype=dtype, safety_checker=None
            )
        except TypeError:  # older diffusers naming
            pipe = AutoPipelineForInpainting.from_pretrained(
                cfg.id, torch_dtype=dtype, safety_checker=None
            )
        if cfg.attention_slicing:
            pipe.enable_attention_slicing()
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, cfg, device)

    def unload(self) -> None:
        del self.pipe
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------- generate

    def generate_one(
        self,
        image: np.ndarray,
        gen_mask: np.ndarray,
        spec: CandidateSpec,
    ) -> tuple[np.ndarray, float]:
        """Generate a single candidate; returns (RGB uint8 image, seconds).

        ``image`` and ``gen_mask`` must already live in generator resolution
        (``cfg.resolution``); the mask marks the region to *repaint*.
        """
        h, w = image.shape[:2]
        res = self.cfg.resolution
        if (h, w) != (res, res):
            raise ValueError(f"expected {res}x{res} input, got {h}x{w}")

        generator = torch.Generator(device=self.device).manual_seed(spec.seed)
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = self.pipe(
                prompt=spec.prompt,
                negative_prompt=self.cfg.negative_prompt,
                image=Image.fromarray(image),
                mask_image=Image.fromarray((gen_mask.astype(np.uint8)) * 255),
                guidance_scale=float(spec.guidance_scale),
                num_inference_steps=int(spec.num_inference_steps),
                height=res,
                width=res,
                generator=generator,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0

        candidate = np.asarray(result.images[0])
        if candidate.shape[:2] != (h, w):
            candidate = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_LANCZOS4)
        candidate = blend_seam(image, candidate, gen_mask, self.cfg.seam_feather_px)
        return candidate, seconds
