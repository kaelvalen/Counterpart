"""counterpart CLI (SPEC.md §10). Commands are added phase by phase."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from counterpart import io as cio
from counterpart.baselines.classical_inpaint import run_baselines
from counterpart.baselines.clip_select import run_clip_scores
from counterpart.baselines.dino_select import run_dino_scores
from counterpart.config import load_config
from counterpart.data.prepare import run_prepare
from counterpart.eval.evaluate import run_evaluate
from counterpart.generate.candidates import run_generate
from counterpart.generate.sd_inpaint import InpaintGenerator, dilate_mask
from counterpart.generate.variants import plan_candidates
from counterpart.score.run import run_score
from counterpart.select.modes import run_modes
from counterpart.select.select import run_select
from counterpart.select.uncertainty import run_uncertainty, uncertainty_heatmap
from counterpart.types import Candidate
from counterpart.viz.panels import contact_sheet, mask_rgb, sample_panel, side_by_side

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


@app.command()
def prepare(
    split: Annotated[str, typer.Option("--split", help="gonogo | train | val | test")] = "gonogo",
    experiment: Annotated[str, typer.Option("--experiment")] = "main",
    limit: Annotated[int | None, typer.Option("--limit", help="max samples to prepare")] = None,
    workers: Annotated[int, typer.Option("--workers")] = 8,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    i_know: Annotated[
        bool,
        typer.Option(
            "--i-know", help="required to touch the test split before Faz 5 (SPEC.md §13)"
        ),
    ] = False,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """ABO image + synthetic damage -> sample cache (Faz 1). Resumable."""
    if split == "test" and not i_know:
        console.print(
            "[red]The test split is gated until Faz 5 (SPEC.md §13). "
            "Pass --i-know if you really mean it.[/red]"
        )
        raise typer.Exit(code=2)
    cfg = load_config(config)
    summary = run_prepare(
        cfg, split, experiment=experiment, limit=limit, workers=workers, overwrite=overwrite
    )
    table = Table(title=f"prepare — {experiment}/{split}")
    table.add_column("key")
    table.add_column("value")
    for key, value in summary.items():
        table.add_row(str(key), str(value))
    console.print(table)


@app.command()
def generate(
    split: Annotated[str, typer.Option("--split")] = "gonogo",
    experiment: Annotated[str, typer.Option("--experiment")] = "main",
    n: Annotated[int | None, typer.Option("--n", help="override generator.n_candidates")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="max samples")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    i_know: Annotated[bool, typer.Option("--i-know", help="test split gate")] = False,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Generate N candidates per prepared sample (Faz 2). Resumable; GT never touched."""
    if split == "test" and not i_know:
        console.print("[red]The test split is gated until Faz 5 (SPEC.md §13).[/red]")
        raise typer.Exit(code=2)
    cfg = load_config(config)
    if n is not None:
        cfg.generator.n_candidates = n
    summary = run_generate(
        cfg, split, experiment=experiment, limit=limit, overwrite=overwrite, verbose=True
    )
    table = Table(title=f"generate — {experiment}/{split}")
    table.add_column("key")
    table.add_column("value")
    for key, value in summary.items():
        table.add_row(str(key), str(value))
    console.print(table)


@app.command()
def evaluate(
    split: Annotated[str, typer.Option("--split")] = "gonogo",
    experiment: Annotated[str, typer.Option("--experiment")] = "main",
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    i_know: Annotated[bool, typer.Option("--i-know", help="test split gate")] = False,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """GT metrics + oracle/first/random/worst comparison; E0 decision on gonogo (Faz 2)."""
    if split == "test" and not i_know:
        console.print("[red]The test split is gated until Faz 5 (SPEC.md §13).[/red]")
        raise typer.Exit(code=2)
    cfg = load_config(config)
    summary = run_evaluate(
        cfg, split, experiment=experiment, limit=limit, overwrite=overwrite, verbose=True
    )
    table = Table(title=f"evaluate — {experiment}/{split}")
    table.add_column("key")
    table.add_column("value")
    for key, value in summary.items():
        if key == "figures":
            continue
        table.add_row(str(key), str(value))
    console.print(table)
    if "figures" in summary:
        for path in summary["figures"]:
            console.print(f"figure: [bold]{path}[/bold]")


@app.command()
def score(
    split: Annotated[str, typer.Option("--split")] = "gonogo",
    experiment: Annotated[str, typer.Option("--experiment")] = "main",
    terms: Annotated[
        str | None, typer.Option("--terms", help="comma-separated subset, e.g. T1,T6")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    workers: Annotated[int, typer.Option("--workers")] = 8,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    i_know: Annotated[bool, typer.Option("--i-know", help="test split gate")] = False,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Score candidates with the classical consistency terms (Faz 3). Cache-only."""
    if split == "test" and not i_know:
        console.print("[red]The test split is gated until Faz 5 (SPEC.md §13).[/red]")
        raise typer.Exit(code=2)
    cfg = load_config(config)
    term_list = [t.strip() for t in terms.split(",")] if terms else None
    summary = run_score(
        cfg,
        split,
        experiment=experiment,
        terms=term_list,
        limit=limit,
        workers=workers,
        overwrite=overwrite,
        verbose=True,
    )
    table = Table(title=f"score — {experiment}/{split}")
    table.add_column("key")
    table.add_column("value")
    for key, value in summary.items():
        table.add_row(str(key), str(value))
    console.print(table)


@app.command()
def baselines(
    split: Annotated[str, typer.Option("--split")] = "gonogo",
    experiment: Annotated[str, typer.Option("--experiment")] = "main",
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    methods: Annotated[
        str | None, typer.Option("--methods", help="comma-separated: telea,ns,patchmatch")
    ] = "telea,ns",
    clip: Annotated[bool, typer.Option("--clip/--no-clip")] = True,
    dino: Annotated[bool, typer.Option("--dino/--no-dino")] = True,
    workers: Annotated[int, typer.Option("--workers")] = 8,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    i_know: Annotated[bool, typer.Option("--i-know", help="test split gate")] = False,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Classical inpainting + CLIP/DINO selectors (Faz 4)."""
    if split == "test" and not i_know:
        console.print("[red]The test split is gated until Faz 5 (SPEC.md §13).[/red]")
        raise typer.Exit(code=2)
    cfg = load_config(config)
    summaries: dict[str, dict] = {}
    if methods:
        cfg.baselines.classical_methods = [m.strip() for m in methods.split(",") if m.strip()]
        summaries["classical"] = run_baselines(
            cfg,
            split,
            experiment=experiment,
            limit=limit,
            workers=workers,
            overwrite=overwrite,
            verbose=True,
        )
    if clip:
        summaries["clip"] = run_clip_scores(
            cfg, split, experiment=experiment, limit=limit, overwrite=overwrite, verbose=True
        )
    if dino:
        summaries["dino"] = run_dino_scores(
            cfg, split, experiment=experiment, limit=limit, overwrite=overwrite, verbose=True
        )
    for name, summary in summaries.items():
        table = Table(title=f"baselines/{name} — {experiment}/{split}")
        table.add_column("key")
        table.add_column("value")
        for key, value in summary.items():
            table.add_row(str(key), str(value))
        console.print(table)


@app.command()
def select(
    split: Annotated[str, typer.Option("--split")] = "gonogo",
    experiment: Annotated[str, typer.Option("--experiment")] = "main",
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    skip_modes: Annotated[bool, typer.Option("--skip-modes")] = False,
    skip_uncertainty: Annotated[bool, typer.Option("--skip-uncertainty")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    i_know: Annotated[bool, typer.Option("--i-know", help="test split gate")] = False,
    config: Annotated[Path, typer.Option("--config", exist_ok=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Selection + modes + uncertainty (Faz 4)."""
    if split == "test" and not i_know:
        console.print("[red]The test split is gated until Faz 5 (SPEC.md §13).[/red]")
        raise typer.Exit(code=2)
    cfg = load_config(config)
    summaries: dict[str, dict] = {}
    summaries["select"] = run_select(
        cfg, split, experiment=experiment, limit=limit, overwrite=overwrite, verbose=True
    )
    if not skip_uncertainty:
        summaries["uncertainty"] = run_uncertainty(
            cfg, split, experiment=experiment, limit=limit, overwrite=overwrite, verbose=True
        )
    if not skip_modes:
        summaries["modes"] = run_modes(
            cfg, split, experiment=experiment, limit=limit, overwrite=overwrite, verbose=True
        )
    for name, summary in summaries.items():
        table = Table(title=f"{name} — {experiment}/{split}")
        table.add_column("key")
        table.add_column("value")
        for key, value in summary.items():
            table.add_row(str(key), str(value))
        console.print(table)


@app.command()
def viz(
    split: Annotated[str, typer.Option("--split")] = "gonogo",
    experiment: Annotated[str, typer.Option("--experiment")] = "main",
    n: Annotated[int, typer.Option("--n", help="number of samples")] = 20,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    kind: Annotated[
        str, typer.Option("--kind", help="qa (data QA strips) | panel (full result panels)")
    ] = "qa",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="output PNG (qa) or directory (panel)"),
    ] = None,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
) -> None:
    """Visualisation: data QA strips or full per-sample result panels (SPEC.md §5.5)."""
    cfg = load_config(config)
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if not sample_ids:
        raise typer.BadParameter(f"no prepared samples under {runs_dir / experiment / split}")

    rng = np.random.default_rng(seed)
    take = min(n, len(sample_ids))
    picked = sorted(rng.choice(len(sample_ids), size=take, replace=False).tolist())

    if kind == "panel":
        out_dir = out or (runs_dir / experiment / "results" / f"panels_{split}")
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for index in picked:
            sample_id = sample_ids[index]
            sample = cio.load_sample(runs_dir, experiment, split, sample_id)
            if not sample.selection_path.exists() or not sample.uncertainty_path.exists():
                continue
            selection = cio.load_json(sample.selection_path)
            modes = (
                cio.load_json(sample.modes_path) if sample.modes_path.exists() else {"modes": []}
            )
            uncertainty = np.load(sample.uncertainty_path)
            damage_mask = cio.read_mask(sample.damage_mask_path)
            best = cio.read_image(sample.candidate_path(int(selection["ours_eq"])))
            representatives = [
                cio.read_image(sample.candidate_path(int(mode["representative_idx"])))
                for mode in modes["modes"][: cfg.select.panel_modes]
            ]
            sample_panel(
                out_dir / f"{sample_id}.png",
                damaged=cio.read_image(sample.damaged_path),
                damage_mask=damage_mask,
                best=best,
                mode_representatives=representatives,
                uncertainty_rgb=uncertainty_heatmap(uncertainty, damage_mask),
                gt=cio.read_image(sample.original_path),
            )
            written += 1
        console.print(f"panels: [bold]{out_dir}[/bold]  ({written} samples)")
        return

    strips, titles = [], []
    for index in picked:
        sample_id = sample_ids[index]
        sample = cio.load_sample(runs_dir, experiment, split, sample_id)
        strips.append(
            side_by_side(
                [
                    cio.read_image(sample.original_path),
                    cio.read_image(sample.damaged_path),
                    mask_rgb(cio.read_mask(sample.damage_mask_path)),
                    mask_rgb(cio.read_mask(sample.gen_mask_path)),
                    mask_rgb(cio.read_mask(sample.object_mask_path)),
                ]
            )
        )
        titles.append(f"{sample_id} [{sample.meta.get('product_type', '?')}]")

    out_path = out or (runs_dir / experiment / "results" / f"qa_{split}.png")
    contact_sheet(strips, titles, out_path, cols=2)
    console.print(f"panel: [bold]{out_path}[/bold]  ({take} samples)")


if __name__ == "__main__":
    app()
