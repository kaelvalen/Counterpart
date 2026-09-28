"""CLIP selector baseline (SPEC.md §5.4, §7).

Scores each candidate by the cosine similarity between its full image and the text
``"a photo of an intact {category}"`` (category from the ABO product type, lowercase).
Writes a ``clip`` column into the sample's ``baseline_scores.parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.generate.candidates import category_text


def run_clip_scores(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    limit: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute CLIP similarity scores for all candidates of a split."""
    import open_clip

    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no samples under {runs_dir / experiment / split}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.baselines.clip_model, pretrained=cfg.baselines.clip_pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(cfg.baselines.clip_model)
    model.eval()

    done = 0
    try:
        with torch.inference_mode():
            for sample_id in sample_ids:
                sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
                records = cio.read_jsonl(sample.candidates_manifest_path)
                if not records:
                    continue
                scores_path = sample.baseline_scores_path
                existing = pd.read_parquet(scores_path) if scores_path.exists() else None
                if (
                    existing is not None
                    and "clip" in existing.columns
                    and not overwrite
                    and set(int(r["idx"]) for r in records) <= set(int(i) for i in existing["idx"])
                ):
                    continue

                category = category_text(sample.meta.get("product_type"))
                prompt = cfg.baselines.clip_prompt.format(category=category)
                text_tokens = tokenizer([prompt]).to(device)
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                tensors = []
                for rec in records:
                    image = Image.fromarray(cio.read_image(sample.candidate_path(int(rec["idx"]))))
                    tensors.append(preprocess(image))
                batch = torch.stack(tensors).to(device)
                image_features = model.encode_image(batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarities = (image_features @ text_features.T).squeeze(1).cpu().numpy()

                frame = pd.DataFrame(
                    {"idx": [int(r["idx"]) for r in records], "clip": similarities.astype(float)}
                )
                if existing is not None and not overwrite:
                    frame = existing.merge(frame, on="idx", how="outer")
                cio.save_parquet(scores_path, frame)
                done += 1
                if verbose:
                    print(f"[{done}] {sample_id}: clip scores for {len(records)} candidates")
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return {"split": split, "experiment": experiment, "samples_scored": done}
