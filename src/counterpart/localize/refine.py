"""Damage-region refinement and convenience heuristics for the demo path.

``concavity_mask`` is the missing-piece helper (SPEC.md §5.2.2/§5.2.3): when a chunk
of an object is *gone*, that area is background now, so the object silhouette has a
concavity exactly where the piece is missing. The convex hull minus the silhouette
therefore yields a damage hint. This is a **demo convenience**, not part of the main
claim: the spec keeps the user-provided hint as the primary source, and this
heuristic degrades on non-convex objects (it will happily flag the gap between a
chair's legs).

The result is a boolean HxW mask on the input image grid.
"""

from __future__ import annotations

import cv2
import numpy as np

from counterpart.config import SegmentCfg
from counterpart.segment.threshold import (
    delta_e76,
    estimate_background_lab,
    fill_holes,
    largest_component,
    object_mask,
    rgb_to_lab,
)


def _significant_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == label
    return keep


def _object_no_fill(image: np.ndarray, seg_cfg: SegmentCfg) -> np.ndarray:
    """Threshold -> close -> largest component; no hole filling (a missing piece can
    look like an interior hole after closing)."""
    lab = rgb_to_lab(image)
    bg = estimate_background_lab(lab, seg_cfg.border_px)
    rough = delta_e76(lab, bg) > seg_cfg.background_delta_e
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (seg_cfg.close_kernel, seg_cfg.close_kernel)
    )
    closed = cv2.morphologyEx(rough.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    obj = largest_component(closed.astype(bool))
    if not obj.any():
        raise ValueError("could not find an object with the background threshold")
    return obj


def symmetry_gap_mask(
    image: np.ndarray,
    seg_cfg: SegmentCfg,
    *,
    min_area_frac: float = 0.004,
    open_kernel: int = 5,
    dilate_px: int = 4,
    max_gap_frac: float = 0.5,
) -> np.ndarray:
    """Damage hint from a mirror test about the object's own bounding-box axis.

    ``mirror(object) & ~object`` highlights where one side has material the other
    side lacks — exactly a missing piece on an approximately mirror-symmetric
    object (vases, mugs, lamps). Strongly asymmetric objects are rejected via
    ``max_gap_frac``.
    """
    obj = _object_no_fill(image, seg_cfg)
    ys, xs = np.nonzero(obj)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    mirrored = np.zeros_like(obj)
    mirrored[y0:y1, x0:x1] = np.fliplr(obj[y0:y1, x0:x1])

    gap = mirrored & ~obj
    frac = float(gap.sum()) / max(float(obj.sum()), 1.0)
    if frac > max_gap_frac:
        raise ValueError(
            f"object is too asymmetric for the mirror hint ({frac:.0%} gap); "
            "please provide a hand mask"
        )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    opened = cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    keep = _significant_components(opened, max(64, int(min_area_frac * float(obj.sum()))))
    if not keep.any():
        raise ValueError("no significant mirror gap found; please provide a hand mask")
    if dilate_px > 0:
        struct = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        keep = cv2.dilate(keep.astype(np.uint8), struct).astype(bool)
    return keep


def missing_piece_mask(
    image: np.ndarray,
    seg_cfg: SegmentCfg,
    *,
    dilate_px: int = 4,
) -> np.ndarray:
    """Union of the mirror-gap and hull-concavity hints (demo convenience).

    The mirror test is precise on symmetric objects but abstains on asymmetric ones;
    the concavity test works on any silhouette but only recovers the part of a large
    cut that the new convex hull does not bridge. Together they cover the common
    missing-piece demos; both are documented as rough hints (SPEC.md §5.2.3).
    """
    candidates: list[np.ndarray] = []
    errors: list[str] = []
    for hint in (symmetry_gap_mask, concavity_mask):
        try:
            candidates.append(hint(image, seg_cfg, dilate_px=dilate_px))
        except ValueError as exc:
            errors.append(f"{hint.__name__}: {exc}")
    if not candidates:
        raise ValueError("no automatic damage hint available — " + "; ".join(errors))
    union = np.zeros_like(candidates[0], dtype=bool)
    for mask in candidates:
        union |= mask
    return union


def concavity_mask(
    image: np.ndarray,
    seg_cfg: SegmentCfg,
    *,
    min_area_frac: float = 0.004,
    open_kernel: int = 5,
    dilate_px: int = 4,
    max_concavity_frac: float = 0.6,
) -> np.ndarray:
    """Damage hint from the concavities of the object silhouette.

    Steps: threshold -> close -> largest component (**no hole filling** — a missing
    piece can look like an interior hole after closing, which would hide it) ->
    convex hull -> hull minus object -> opening (drop thin slivers) -> keep large
    components -> dilate.

    Caveat: natural concavities (a vase's shoulders, a chair's leg gap) are also
    flagged, and a deep/wide cut is only partially recovered (the new convex hull
    bridges most of it). This is a rough demo hint, not a detector (SPEC.md §5.2.3);
    prefer a hand mask — or :func:`symmetry_gap_mask` on symmetric objects.
    ``max_concavity_frac`` guards against degenerate silhouettes whose hull is mostly
    empty.
    """
    obj = _object_no_fill(image, seg_cfg)

    contours, _ = cv2.findContours(obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("object mask has no contours")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_mask = np.zeros(obj.shape, dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 1)
    hull_mask = hull_mask.astype(bool)

    concave = hull_mask & ~obj
    area_frac = float(concave.sum()) / max(float(obj.sum()), 1.0)
    if area_frac > max_concavity_frac:
        raise ValueError(
            f"silhouette is too non-convex for the concavity heuristic "
            f"({area_frac:.0%} of the hull is empty); please provide a hand mask"
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    opened = cv2.morphologyEx(concave.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    # keep only components large enough to be a missing piece, not edge noise
    min_area = max(64, int(min_area_frac * float(obj.sum())))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened.astype(np.uint8), 8)
    keep = np.zeros_like(opened)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == label
    if not keep.any():
        raise ValueError("no significant concavity found; please provide a hand mask")

    if dilate_px > 0:
        struct = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        keep = cv2.dilate(keep.astype(np.uint8), struct).astype(bool)
    return keep


def should_be_object_mask(image: np.ndarray, seg_cfg: SegmentCfg) -> np.ndarray:
    """The object's *expected* silhouette (hull) — used as ``object_mask`` for demo
    scoring, so terms that reason about the silhouette see the pre-damage extent."""
    obj = object_mask(image, seg_cfg)
    contours, _ = cv2.findContours(obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("object mask has no contours")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_mask = np.zeros(obj.shape, dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 1)
    return fill_holes(largest_component(hull_mask.astype(bool)))
