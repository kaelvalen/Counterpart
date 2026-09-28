"""Classical inpainting baselines: Telea, Navier-Stokes, (optional) PatchMatch.

Each method produces one completion per sample, cached as
``<sample>/baselines/<method>.png`` (plus a timing record). These are compared
against the generative + selection pipeline in E1 (H3).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from counterpart import io as cio
from counterpart.config import Cfg

CV2_METHODS = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


def inpaint_cv2(image: np.ndarray, mask: np.ndarray, method: str, radius: int = 3) -> np.ndarray:
    """OpenCV inpainting on RGB uint8; ``mask`` marks pixels to replace."""
    if method not in CV2_METHODS:
        raise ValueError(f"unknown cv2 method: {method}")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, (mask.astype(np.uint8)) * 255, radius, CV2_METHODS[method])
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def inpaint_patchmatch(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """PyPatchMatch if installed (SPEC.md §7); returns None when unavailable."""
    try:
        import pypatchmatch  # type: ignore
    except ImportError:
        return None
    width, height = image.shape[1], image.shape[0]
    result = pypatchmatch.inpaint(
        width,
        height,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        ((mask.astype(np.uint8)) * 255),
    )
    out = np.asarray(result[1]) if isinstance(result, tuple) else np.asarray(result)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def baseline_dir(runs_dir: Path, experiment: str, split: str, sample_id: str) -> Path:
    return Path(runs_dir) / experiment / split / sample_id / "baselines"


def run_baselines_sample(
    cfg: Cfg, experiment: str, split: str, sample_id: str, *, overwrite: bool = False
) -> dict[str, Any]:
    """Compute the configured classical baselines for one sample (resumable)."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample = cio.load_sample(runs_dir, experiment, split, sample_id, gt_hidden=True)
    out_dir = baseline_dir(runs_dir, experiment, split, sample_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    damaged = cio.read_image(sample.damaged_path)
    gen_mask = cio.read_mask(sample.gen_mask_path)  # classical methods use the same region

    timings: dict[str, float] = {}
    for method in cfg.baselines.classical_methods:
        path = out_dir / f"{method}.png"
        if path.exists() and not overwrite:
            continue
        start = time.perf_counter()
        if method in CV2_METHODS:
            filled = inpaint_cv2(damaged, gen_mask, method)
        elif method == "patchmatch":
            filled = inpaint_patchmatch(damaged, gen_mask)
            if filled is None:
                timings[method] = float("nan")
                continue
        else:
            raise ValueError(f"unknown classical method: {method}")
        filled = np.where(gen_mask[..., None], filled, damaged)  # exact outside the mask
        cio.write_image(path, filled)
        timings[method] = round(time.perf_counter() - start, 4)

    if timings:
        cio.save_json(out_dir / "timings.json", timings)
    return {"sample_id": sample_id, "methods": timings or "cached"}


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = Cfg.model_validate(payload["cfg"])
    return run_baselines_sample(
        cfg,
        payload["experiment"],
        payload["split"],
        payload["sample_id"],
        overwrite=payload["overwrite"],
    )


def run_baselines(
    cfg: Cfg,
    split: str,
    *,
    experiment: str = "main",
    limit: int | None = None,
    workers: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run classical baselines over a split (parallel over samples)."""
    runs_dir = Path(cfg.project.paths.runs_dir)
    sample_ids = cio.iter_sample_ids(runs_dir, experiment, split)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError(f"no prepared samples under {runs_dir / experiment / split}")

    payloads = [
        {
            "sample_id": sample_id,
            "experiment": experiment,
            "split": split,
            "overwrite": overwrite,
            "cfg": cfg.model_dump(mode="json"),
        }
        for sample_id in sample_ids
    ]
    counts = {"ok": 0, "error": 0}
    errors: list[str] = []
    if workers <= 1:
        for payload in payloads:
            result = _worker(payload)
            counts["ok"] += 1
            if verbose:
                print(f"{result['sample_id']}: {result['methods']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_worker, payloads):
                counts["ok"] += 1
                if verbose:
                    done = counts["ok"]
                    print(f"[{done}/{len(payloads)}] {result['sample_id']}: {result['methods']}")
    return {
        "split": split,
        "experiment": experiment,
        "samples": len(sample_ids),
        **counts,
        "errors": errors,
    }
