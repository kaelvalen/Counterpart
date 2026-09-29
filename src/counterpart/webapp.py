"""Optional Gradio demo (SPEC.md §2, Faz 6).

Image + mask in, best completion (classical scorer pick) + uncertainty + candidate
stripes out. The heavy stages run on the local GPU; the app reuses the *same*
scoring pipeline as the CLI by materialising a temporary sample directory in the
project cache layout.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from counterpart import io as cio
from counterpart.config import Cfg, load_config
from counterpart.generate.sd_inpaint import InpaintGenerator, dilate_mask
from counterpart.generate.variants import plan_candidates
from counterpart.score.run import score_sample
from counterpart.select.uncertainty import _dispersion, uncertainty_heatmap
from counterpart.viz.panels import mask_rgb, side_by_side


def _fit_square(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop to square and resize to ``size`` (demo-grade preprocessing)."""
    import cv2

    h, w = image.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    crop = image[y0 : y0 + side, x0 : x0 + side]
    crop_mask = mask[y0 : y0 + side, x0 : x0 + side]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    image_r = cv2.resize(crop, (size, size), interpolation=interp)
    mask_r = cv2.resize(
        crop_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return image_r, mask_r


def reconstruct(
    cfg: Cfg,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    n: int = 8,
    prompt: str | None = None,
    category: str | None = None,
    object_mask_override: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Run generate -> score -> select on one image+mask; returns panel-ready arrays."""
    from counterpart.segment.threshold import object_mask

    gen_cfg = cfg.generator.model_copy(update={"n_candidates": n})
    if prompt:
        gen_cfg = gen_cfg.model_copy(update={"prompts": [prompt]})

    size = gen_cfg.resolution
    damaged, damage_mask = _fit_square(image, mask, size)
    gen_mask = dilate_mask(damage_mask, cfg.damage.gen_mask_dilate_px)
    if object_mask_override is not None:
        _, obj = _fit_square(image, object_mask_override, size)
    else:
        obj = object_mask(damaged, cfg.segment)
        if not obj.any():
            obj = ~damage_mask
    obj = obj | damage_mask  # the object mask must cover the damage region

    # cache identity: every generator-relevant input must invalidate stale candidates
    import json

    key = json.dumps(
        {
            "image": hashlib.sha256(damaged.tobytes()).hexdigest()[:16],
            "mask": hashlib.sha256(damage_mask.tobytes()).hexdigest()[:16],
            "prompt": prompt or "",
            "negative": gen_cfg.negative_prompt,
            "guidance": gen_cfg.guidance_scales,
            "steps": gen_cfg.num_inference_steps,
            "model": gen_cfg.id,
        },
        sort_keys=True,
    )
    tag = hashlib.sha256(key.encode()).hexdigest()[:12]
    runs_dir = Path(cfg.project.paths.runs_dir)
    root = runs_dir / "webapp" / "demo" / tag
    candidates_dir = root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    cio.write_image(root / "damaged.png", damaged)
    cio.write_mask(root / "damage_mask.png", damage_mask)
    cio.write_mask(root / "gen_mask.png", gen_mask)
    cio.write_mask(root / "object_mask.png", obj)
    cio.save_json(root / "meta.json", {"sample_id": tag, "split": "demo", "source": "webapp"})

    existing = {rec["idx"] for rec in cio.read_jsonl(root / "candidates.jsonl")}
    specs = plan_candidates(n, gen_cfg, sample_id=tag, category=category)
    todo = [
        s
        for s in specs
        if s.idx not in existing or not (candidates_dir / f"cand_{s.idx:03d}.png").exists()
    ]
    if todo:
        generator = InpaintGenerator.load(gen_cfg)
        try:
            for spec in todo:
                candidate, seconds = generator.generate_one(damaged, gen_mask, spec)
                cio.write_image(candidates_dir / f"cand_{spec.idx:03d}.png", candidate)
                cio.append_jsonl(
                    root / "candidates.jsonl",
                    [
                        {
                            "idx": spec.idx,
                            "path": str(candidates_dir / f"cand_{spec.idx:03d}.png"),
                            "seed": spec.seed,
                            "prompt": spec.prompt,
                            "guidance_scale": spec.guidance_scale,
                            "num_inference_steps": spec.num_inference_steps,
                            "generator_id": gen_cfg.id,
                            "seconds": round(seconds, 3),
                        }
                    ],
                )
        finally:
            generator.unload()

    temp_cfg = cfg.model_copy(deep=True)
    temp_cfg.project.paths.runs_dir = runs_dir  # type: ignore[assignment]
    frame = score_sample(temp_cfg, "webapp", "demo", tag)
    best_idx = int(frame.loc[frame["combined_A"].idxmax(), "idx"])

    candidates = [cio.read_image(candidates_dir / f"cand_{int(i):03d}.png") for i in frame["idx"]]
    uncertainty = _dispersion(candidates)
    uncertainty[~damage_mask] = 0.0

    strip = side_by_side([damaged, mask_rgb(damage_mask), *candidates[:8]])
    return {
        "damaged": damaged,
        "best": candidates[list(frame["idx"]).index(best_idx)],
        "uncertainty": uncertainty_heatmap(uncertainty, damage_mask),
        "candidates": strip,
        "mean_uncertainty": float(uncertainty.mean()),
        "best_score": float(frame["combined_A"].max()),
    }


def build_demo(config: str = "configs/default.yaml"):
    """Build the Gradio interface (imports gradio lazily)."""
    import gradio as gr

    cfg = load_config(config)

    def run(image, mask, prompt, n_candidates):
        if image is None or mask is None:
            raise gr.Error("Please provide both an image and a damage mask.")
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_bool = mask_array > 127
        if not mask_bool.any():
            raise gr.Error("The mask is empty — paint the damaged region in white.")
        result = reconstruct(
            cfg, np.asarray(image), mask_bool, n=int(n_candidates), prompt=prompt or None
        )
        return (
            result["best"],
            result["uncertainty"],
            result["candidates"],
            f"mean pixel uncertainty: {result['mean_uncertainty']:.2f} | "
            f"best combined score: {result['best_score']:.3f}",
        )

    with gr.Blocks(title="counterpart — damaged object reconstruction") as demo:
        gr.Markdown(
            "## counterpart\n"
            "Frozen diffusion inpainting produces N candidates; the classical "
            "consistency terms pick the most plausible one and map the uncertainty."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Damaged object", type="numpy")
                mask_input = gr.Image(label="Damage mask (white = damaged)", type="numpy")
                prompt_input = gr.Textbox(label="Prompt (optional)")
                n_input = gr.Slider(2, 16, value=8, step=1, label="Candidates")
                run_button = gr.Button("Reconstruct", variant="primary")
            with gr.Column():
                best_output = gr.Image(label="Best completion (mode A)")
                uncertainty_output = gr.Image(label="Pixel uncertainty")
                status = gr.Textbox(label="Status")
        candidates_output = gr.Image(label="Damaged | mask | candidates")
        run_button.click(
            run,
            inputs=[image_input, mask_input, prompt_input, n_input],
            outputs=[best_output, uncertainty_output, candidates_output, status],
        )
    return demo
