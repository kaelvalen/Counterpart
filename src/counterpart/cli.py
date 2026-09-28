"""counterpart CLI (SPEC.md §10). Commands are added phase by phase."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer
from rich.console import Console

from counterpart import io as cio
from counterpart.config import load_config
from counterpart.generate.sd_inpaint import InpaintGenerator, dilate_mask
from counterpart.generate.variants import plan_candidates
from counterpart.types import Candidate
from counterpart.viz.panels import contact_sheet

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def main() -> None:
    """counterpart — hypothesis-based reconstruction of damaged objects."""


def _fit_square(
    image: np.ndarray, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Center-crop image+mask to a square, then resize to ``size``.

    Returns (image, mask, (x0, y0, side)) where the crop box is in original pixels.
    (Proper object-bbox crop policy arrives with the data layer; demo images are
    assumed roughly centered and square already.)
    """
    h, w = image.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    crop = image[y0 : y0 + side, x0 : x0 + side]
    m = mask[y0 : y0 + side, x0 : x0 + side]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    image_r = cv2.resize(crop, (size, size), interpolation=interp)
    mask_r = cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(
        bool
    )
    return image_r, mask_r, (x0, y0, side)


@app.command()
def demo(
    image: Annotated[
        Path, typer.Option("--image", exists=True, dir_okay=False, help="Damaged object image")
    ],
    mask: Annotated[
        Path,
        typer.Option(
            "--mask",
            exists=True,
            dir_okay=False,
            help="Damage hint mask (0/255 PNG, white = damaged)",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory")] = Path("runs/demo"),
    n: Annotated[
        int | None,
        typer.Option("--n", help="Candidate count (default: config generator.n_candidates)"),
    ] = None,
    steps: Annotated[int | None, typer.Option("--steps", help="Inference steps override")] = None,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", help="Prompt override (default: config prompt set)"),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option("--category", help="Category used to fill prompt templates"),
    ] = None,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Regenerate candidates that already exist"),
    ] = False,
) -> None:
    """Generate N candidates for a single image + hand mask.

    Faz 0 kabul testi ve E8 açık-dünya yolu (SPEC.md §11, §9).
    """
    cfg = load_config(config)
    gen_cfg = cfg.generator
    if n is not None:
        gen_cfg.n_candidates = n
    if steps is not None:
        gen_cfg.num_inference_steps = steps
    if prompt is not None:
        gen_cfg.prompts = [prompt]

    image_raw = cio.read_image(image)
    mask_raw = cio.read_mask(mask)
    if not mask_raw.any():
        raise typer.BadParameter("mask is empty — nothing to reconstruct")

    sample_dir = out / image.stem
    candidates_dir = sample_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    size = gen_cfg.resolution
    damaged, damage_mask, _crop = _fit_square(image_raw, mask_raw, size)
    gen_mask = dilate_mask(damage_mask, cfg.damage.gen_mask_dilate_px)

    cio.write_image(sample_dir / "damaged.png", damaged)
    cio.write_mask(sample_dir / "damage_mask.png", damage_mask)
    cio.write_mask(sample_dir / "gen_mask.png", gen_mask)

    specs = plan_candidates(gen_cfg.n_candidates, gen_cfg, sample_id=image.stem, category=category)
    existing = {rec["idx"]: rec for rec in cio.read_jsonl(sample_dir / "candidates.jsonl")}

    todo = [
        s
        for s in specs
        if overwrite
        or s.idx not in existing
        or not (candidates_dir / f"cand_{s.idx:03d}.png").exists()
    ]
    console.print(
        f"[bold]{sample_dir}[/bold] — {len(specs)} candidates planned, "
        f"{len(specs) - len(todo)} cached, {len(todo)} to generate "
        f"(steps={gen_cfg.num_inference_steps}, model={gen_cfg.id})"
    )

    if todo:
        generator = InpaintGenerator.load(gen_cfg)
        try:
            for spec in todo:
                cand_img, seconds = generator.generate_one(damaged, gen_mask, spec)
                cio.write_image(candidates_dir / f"cand_{spec.idx:03d}.png", cand_img)
                record = Candidate(
                    idx=spec.idx,
                    path=candidates_dir / f"cand_{spec.idx:03d}.png",
                    seed=spec.seed,
                    prompt=spec.prompt,
                    guidance_scale=spec.guidance_scale,
                    num_inference_steps=spec.num_inference_steps,
                    generator_id=gen_cfg.id,
                    seconds=round(seconds, 3),
                )
                existing[spec.idx] = record.to_json()
                cio.append_jsonl(sample_dir / "candidates.jsonl", [record.to_json()])
                console.print(
                    f"  cand {spec.idx:03d}  {seconds:5.2f}s  "
                    f"g={spec.guidance_scale}  seed={spec.seed}"
                )
        finally:
            generator.unload()

    # Refresh contact sheet from all candidates on disk.
    images, titles = [], []
    for idx in sorted(existing):
        path = candidates_dir / f"cand_{idx:03d}.png"
        if path.exists():
            images.append(cio.read_image(path))
            rec = existing[idx]
            titles.append(f"{idx:03d} g={rec.get('guidance_scale')} s={rec.get('seed') % 100000}")
    if images:
        contact_sheet(images, titles, sample_dir / "panel.png")
        console.print(f"panel: [bold]{sample_dir / 'panel.png'}[/bold]")

    shutil.copyfile(image, sample_dir / "input.png")
    console.print("done.")


if __name__ == "__main__":
    app()
