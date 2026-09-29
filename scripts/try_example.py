#!/usr/bin/env python
"""Try counterpart on a single open-world image (E8 demo convenience).

Usage:
    uv run python scripts/try_example.py --image demo/inputs/kirik_vazo.jpg \
        --prompt "a complete white ceramic vase, product photo, white background" --n 16

Without ``--mask`` the damage hint is derived from the silhouette concavities
(missing-piece heuristic, localize/refine.py). Outputs land in
``runs/demo/<name>/``: damaged.png, mask.png, best.png, uncertainty.png,
candidates.png plus a numeric summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from counterpart import io as cio
from counterpart.config import load_config
from counterpart.localize.refine import concavity_mask, should_be_object_mask
from counterpart.viz.panels import side_by_side
from counterpart.webapp import reconstruct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None, help="hand mask PNG (optional)")
    parser.add_argument("--name", default=None, help="output slug (default: image stem)")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="runs/demo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_path = Path(args.image)
    name = args.name or image_path.stem
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cio.read_image(image_path)
    if args.mask:
        mask = cio.read_mask(Path(args.mask))
        hint_source = "hand mask"
        obj_override = None
    else:
        mask = concavity_mask(image, cfg.segment)
        obj_override = should_be_object_mask(image, cfg.segment)
        hint_source = "concavity heuristic"
    print(
        f"{name}: {image.shape[1]}x{image.shape[0]} | hint: {hint_source} | mask px: {int(mask.sum())}"
    )

    result = reconstruct(
        cfg, image, mask, n=args.n, prompt=args.prompt, object_mask_override=obj_override
    )

    cio.write_image(out_dir / "damaged.png", result["damaged"])  # type: ignore[arg-type]
    cio.write_mask(out_dir / "mask.png", mask)
    cio.write_image(out_dir / "best.png", result["best"])  # type: ignore[arg-type]
    cio.write_image(out_dir / "uncertainty.png", result["uncertainty"])  # type: ignore[arg-type]
    cio.write_image(out_dir / "candidates.png", result["candidates"])  # type: ignore[arg-type]
    combined = side_by_side(
        [
            result["damaged"],  # type: ignore[list-item]
            result["best"],  # type: ignore[list-item]
            result["uncertainty"],  # type: ignore[list-item]
            np.asarray(result["candidates"]),  # type: ignore[arg-type]
        ]
    )
    cio.write_image(out_dir / "summary.png", combined)
    print(
        f"best combined score: {float(result['best_score']):.3f} | "
        f"mean uncertainty: {float(result['mean_uncertainty']):.2f}\n"
        f"outputs: {out_dir}/ (damaged, best, uncertainty, candidates, summary)"
    )


if __name__ == "__main__":
    main()
