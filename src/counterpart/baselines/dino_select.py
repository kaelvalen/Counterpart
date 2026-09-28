"""DINOv2 selector baseline (SPEC.md §5.4, §7).

Two signals, combined with configurable weights:

* CLS similarity between a candidate and the *visible* reference image (the damaged
  image with the damage region replaced by neutral grey, so only V and the
  surrounding context are seen);
* nearest-neighbour similarity: mean over the candidate's damage-region patch tokens
  of the maximum cosine similarity to the reference's visible-object patch tokens.

Writes a ``dino`` column into the sample's ``baseline_scores.parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from counterpart import io as cio
from counterpart.config import Cfg


def _token_grid(processor, model, image: Image.Image, device: str) -> tuple[torch.Tensor, int, int]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    tokens = outputs.last_hidden_state  # (1, 1 + H/patch * W/patch, D)
    patch_size = int(getattr(model.config, "patch_size", 14))
    gh = inputs["pixel_values"].shape[-2] // patch_size
    gw = inputs["pixel_values"].shape[-1] // patch_size
    return tokens, gh, gw


def _cls_and_patches(tokens: torch.Tensor, gh: int, gw: int) -> tuple[torch.Tensor, torch.Tensor]:
    cls = tokens[:, 0]  # (1, D)
    patches = tokens[:, 1 : 1 + gh * gw].reshape(gh, gw, -1)
    return cls, patches


def _mask_to_grid(mask: np.ndarray, gh: int, gw: int, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(tensor.to(device), (gh, gw))[0, 0]
    return pooled > 0.5


def run_dino_scores(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    limit: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute DINOv2 selector scores for all candidates of a split."""
    from transformers import AutoImageProcessor, AutoModel

    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no samples under {runs_dir / experiment / split}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(cfg.baselines.dino_model)
    model = AutoModel.from_pretrained(cfg.baselines.dino_model).to(device).eval()

    done = 0
    try:
        for sample_id in sample_ids:
            sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
            records = cio.read_jsonl(sample.candidates_manifest_path)
            if not records:
                continue
            scores_path = sample.baseline_scores_path
            existing = pd.read_parquet(scores_path) if scores_path.exists() else None
            if (
                existing is not None
                and "dino" in existing.columns
                and not overwrite
                and set(int(r["idx"]) for r in records) <= set(int(i) for i in existing["idx"])
            ):
                continue

            damaged = cio.read_image(sample.damaged_path)
            damage_mask = cio.read_mask(sample.damage_mask_path)
            visible = cio.read_mask(sample.object_mask_path) & ~damage_mask

            reference = damaged.copy()
            reference[damage_mask] = cfg.baselines.neutral_gray
            ref_tokens, gh, gw = _token_grid(processor, model, Image.fromarray(reference), device)
            ref_cls, ref_patches = _cls_and_patches(ref_tokens, gh, gw)
            visible_grid = _mask_to_grid(visible, gh, gw, device)
            ref_patch_vectors = ref_patches[visible_grid]
            if ref_patch_vectors.numel() == 0:
                ref_patch_vectors = ref_patches.reshape(-1, ref_patches.shape[-1])

            damage_grid = _mask_to_grid(damage_mask, gh, gw, device)

            cls_sims, nn_sims = [], []
            for rec in records:
                candidate = Image.fromarray(cio.read_image(sample.candidate_path(int(rec["idx"]))))
                tokens, gh_c, gw_c = _token_grid(processor, model, candidate, device)
                cls, patches = _cls_and_patches(tokens, gh_c, gw_c)
                cls_sims.append(float(F.cosine_similarity(cls, ref_cls, dim=-1).item()))

                patch_vectors = patches.reshape(-1, patches.shape[-1])
                selected = patches[damage_grid]
                if selected.numel() == 0:
                    selected = patch_vectors
                sim = F.normalize(selected, dim=-1) @ F.normalize(ref_patch_vectors, dim=-1).T
                nn_sims.append(float(sim.max(dim=1).values.mean().item()))

            score = cfg.baselines.dino_cls_weight * np.asarray(
                cls_sims
            ) + cfg.baselines.dino_nn_weight * np.asarray(nn_sims)
            frame = pd.DataFrame(
                {"idx": [int(r["idx"]) for r in records], "dino": score.astype(float)}
            )
            if existing is not None and not overwrite:
                frame = existing.merge(frame, on="idx", how="outer")
            cio.save_parquet(scores_path, frame)
            done += 1
            if verbose:
                print(f"[{done}] {sample_id}: dino scores for {len(records)} candidates")
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return {"split": split, "experiment": experiment, "samples_scored": done}
