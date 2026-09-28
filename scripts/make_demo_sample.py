#!/usr/bin/env python
"""Create a demo sample from a raw product photo (ABO or Kael's own photos).

Steps: estimate object silhouette by background thresholding -> square crop around
the object bbox + 15% margin -> 512x512 -> carve one synthetic damage instance.

Outputs to ``--out`` (default ``demo/inputs/``):
  <name>_damaged.png   512x512 damaged image (demo input)
  <name>_mask.png      512x512 damage mask (white = damaged, demo input)
  <name>_original.png  512x512 intact reference (report/qualitative only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from counterpart import io as cio
from counterpart.config import load_config
from counterpart.data.damage import synthesize_damage
from counterpart.data.masks import sample_damage


def object_mask_from_background(image: np.ndarray, l_thresh: int = 240) -> np.ndarray:
    """Threshold near-white background, keep the largest component, fill holes."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    obj = (~(gray > l_thresh)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
    if n <= 1:
        raise RuntimeError("no object found with the given background threshold")
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    obj = (labels == idx).astype(np.uint8)

    # fill interior holes via floodFill of the complement from a corner
    ff = obj.copy()
    h, w = obj.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 1)
    obj = obj | (1 - ff)
    return obj.astype(bool)


def square_crop_box(object_mask: np.ndarray, margin: float = 0.15) -> tuple[int, int, int]:
    """Square box (x0, y0, side) around the object bbox with a relative margin."""
    ys, xs = np.nonzero(object_mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = object_mask.shape
    side = int(round(max(x1 - x0 + 1, y1 - y0 + 1) * (1.0 + 2 * margin)))
    side = min(side, h, w)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return x0, y0, side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="raw product photo")
    parser.add_argument("--name", required=True, help="output sample name")
    parser.add_argument("--out", default="demo/inputs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--damage-type", default="chip", choices=["chip", "hole", "crack"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    image = cio.read_image(Path(args.image))
    obj_full = object_mask_from_background(image)
    x0, y0, side = square_crop_box(obj_full, margin=0.15)

    crop = image[y0 : y0 + side, x0 : x0 + side]
    obj_crop = obj_full[y0 : y0 + side, x0 : x0 + side]
    interp = cv2.INTER_AREA if side > args.size else cv2.INTER_LINEAR
    image_512 = cv2.resize(crop, (args.size, args.size), interpolation=interp)
    obj_512 = cv2.resize(
        obj_crop.astype(np.uint8), (args.size, args.size), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    damage_mask, info = sample_damage(obj_512, cfg.damage, rng, forced_type=args.damage_type)
    damaged = synthesize_damage(image_512, damage_mask, info.damage_type, cfg.damage, rng)

    out = Path(args.out)
    cio.write_image(out / f"{args.name}_damaged.png", damaged)
    cio.write_mask(out / f"{args.name}_mask.png", damage_mask)
    cio.write_image(out / f"{args.name}_original.png", image_512)
    print(
        f"{args.name}: {info.damage_type}, area_frac={info.area_frac:.3f}, "
        f"attempts={info.attempts} -> {out}/{args.name}_damaged.png"
    )


if __name__ == "__main__":
    main()
