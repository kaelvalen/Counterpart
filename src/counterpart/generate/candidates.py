"""Candidate generation over prepared samples (SPEC.md §5.3, §10 `generate` command).

Resumable: existing candidate PNGs + manifest rows are reused; only missing indices
are generated. Ground truth is never loaded here (``gt_hidden=True`` sample view).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from counterpart import io as cio
from counterpart.config import Cfg
from counterpart.generate.sd_inpaint import InpaintGenerator
from counterpart.generate.variants import plan_candidates
from counterpart.types import Candidate


def category_text(product_type: str | None) -> str:
    """ABO product_type -> natural-language category for prompt templates."""
    if not product_type:
        return "object"
    return product_type.lower().replace("_", " ")


def generate_for_sample(
    generator: InpaintGenerator,
    cfg: Cfg,
    experiment: str,
    split: str,
    sample_id: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the missing candidates for one sample; returns a per-sample summary."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
    damaged = cio.read_image(sample.damaged_path)
    gen_mask = cio.read_mask(sample.gen_mask_path)

    specs = plan_candidates(
        cfg.generator.n_candidates,
        cfg.generator,
        sample_id,
        category=category_text(sample.meta.get("product_type")),
    )
    existing = {rec["idx"]: rec for rec in cio.read_jsonl(sample.candidates_manifest_path)}
    todo = [
        spec
        for spec in specs
        if overwrite or spec.idx not in existing or not sample.candidate_path(spec.idx).exists()
    ]

    generated = 0
    seconds_total = 0.0
    for spec in todo:
        candidate_image, seconds = generator.generate_one(damaged, gen_mask, spec)
        cio.write_image(sample.candidate_path(spec.idx), candidate_image)
        record = Candidate(
            idx=spec.idx,
            path=sample.candidate_path(spec.idx),
            seed=spec.seed,
            prompt=spec.prompt,
            guidance_scale=spec.guidance_scale,
            num_inference_steps=spec.num_inference_steps,
            generator_id=cfg.generator.id,
            seconds=round(seconds, 3),
        )
        cio.append_jsonl(sample.candidates_manifest_path, [record.to_json()])
        generated += 1
        seconds_total += seconds

    return {
        "sample_id": sample_id,
        "planned": len(specs),
        "generated": generated,
        "skipped": len(specs) - len(todo),
        "seconds": round(seconds_total, 2),
    }


def run_generate(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    limit: int | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate candidates for every sample of a split (resumable)."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no prepared samples under {runs_dir / experiment / split}")

    generator = InpaintGenerator.load(cfg.generator)
    started = time.perf_counter()
    generated = skipped = 0
    try:
        for i, sample_id in enumerate(sample_ids, start=1):
            result = generate_for_sample(
                generator, cfg, experiment, split, sample_id, overwrite=overwrite
            )
            generated += result["generated"]
            skipped += result["skipped"]
            if verbose:
                print(
                    f"[{i}/{len(sample_ids)}] {sample_id}: "
                    f"+{result['generated']} (skip {result['skipped']}) "
                    f"{result['seconds']:.1f}s"
                )
    finally:
        generator.unload()

    return {
        "split": split,
        "experiment": experiment,
        "samples": len(sample_ids),
        "candidates_generated": generated,
        "candidates_skipped": skipped,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
