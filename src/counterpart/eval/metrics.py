"""Ground-truth reconstruction metrics (SPEC.md §8.1).

All functions take RGB uint8 HxWx3 arrays and bool HxW masks. The evaluation region
is the damage mask bbox + margin for LPIPS, and the damage mask itself (or its
boundary rings) for pixel metrics.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch

from counterpart.config import EvalCfg, SegmentCfg
from counterpart.segment.threshold import delta_e76, object_mask, rgb_to_lab


@lru_cache(maxsize=2)
def get_lpips(net: str = "alex") -> Any:
    """Lazily loaded LPIPS model (weights cached on disk by the package)."""
    import lpips

    model = lpips.LPIPS(net=net, verbose=False)
    if torch.cuda.is_available():
        model = model.cuda()
    return model.eval()


def _to_lpips_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def crop_box_for_mask(
    mask: np.ndarray, margin: int, min_side: int = 0
) -> tuple[int, int, int, int]:
    """Bounding box of ``mask`` + margin, optionally grown to at least ``min_side``."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    y0, y1 = int(ys.min()) - margin, int(ys.max()) + margin + 1
    x0, x1 = int(xs.min()) - margin, int(xs.max()) + margin + 1

    h, w = mask.shape
    # grow symmetrically to min_side where the image allows it
    for _ in range(2):
        if y1 - y0 < min_side:
            need = min_side - (y1 - y0)
            y0, y1 = y0 - need // 2, y1 + need - need // 2
        if x1 - x0 < min_side:
            need = min_side - (x1 - x0)
            x0, x1 = x0 - need // 2, x1 + need - need // 2
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    # final clamp: if the image is smaller than min_side, take the whole image
    return y0, y1, x0, x1


def lpips_score(
    gt: np.ndarray, candidate: np.ndarray, damage_mask: np.ndarray, cfg: EvalCfg
) -> float:
    """LPIPS over the damage-mask bbox + margin (lower is better)."""
    y0, y1, x0, x1 = crop_box_for_mask(damage_mask, cfg.lpips_crop_margin, cfg.lpips_min_crop)
    a = _to_lpips_tensor(gt[y0:y1, x0:x1])
    b = _to_lpips_tensor(candidate[y0:y1, x0:x1])
    with torch.inference_mode():
        value = get_lpips(cfg.lpips_net)(a, b)
    return float(value.item())


def masked_psnr(gt: np.ndarray, candidate: np.ndarray, damage_mask: np.ndarray) -> float:
    """PSNR over the damage mask pixels (RGB, 0-1 scale)."""
    a = gt[damage_mask].astype(np.float64) / 255.0
    b = candidate[damage_mask].astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def masked_ssim(gt: np.ndarray, candidate: np.ndarray, damage_mask: np.ndarray) -> float:
    """Mean SSIM over the damage mask (windowed map, averaged inside the mask)."""
    from skimage.metrics import structural_similarity as _ssim

    # the mask can be tiny; pad the crop a little so the window fits
    y0, y1, x0, x1 = crop_box_for_mask(damage_mask, margin=8, min_side=32)
    gray_a = cv2.cvtColor(gt[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(candidate[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    _, ssim_map = _ssim(gray_a, gray_b, full=True, data_range=255)
    return float(ssim_map[damage_mask[y0:y1, x0:x1]].mean())


def _ring_masks(damage_mask: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    dil = cv2.dilate(damage_mask.astype(np.uint8), kernel).astype(bool)
    ero = cv2.erode(damage_mask.astype(np.uint8), kernel).astype(bool)
    ring_out = dil & ~damage_mask
    ring_in = damage_mask & ~ero
    return ring_in, ring_out


def boundary_delta_e(
    gt: np.ndarray, candidate: np.ndarray, damage_mask: np.ndarray, ring_px: int
) -> float:
    """Mean ΔE between GT and candidate on the rings just inside/outside the mask."""
    ring_in, ring_out = _ring_masks(damage_mask, ring_px)
    band = ring_in | ring_out
    if not band.any():
        return float("nan")
    lab_gt = rgb_to_lab(gt)
    lab_cand = rgb_to_lab(candidate)
    return float(delta_e76(lab_gt[band], lab_cand[band]).mean())


def silhouette_iou(
    candidate: np.ndarray,
    gt_object_mask: np.ndarray,
    damage_mask: np.ndarray,
    seg_cfg: SegmentCfg,
    margin: int = 16,
) -> float:
    """IoU of GT vs candidate object silhouettes inside the damage bbox + margin."""
    y0, y1, x0, x1 = crop_box_for_mask(damage_mask, margin)
    cand_obj = object_mask(candidate, seg_cfg)
    a = cand_obj[y0:y1, x0:x1]
    b = gt_object_mask[y0:y1, x0:x1]
    union = int((a | b).sum())
    if union == 0:
        return float("nan")
    return float((a & b).sum()) / union


def reconstruction_metrics(
    gt: np.ndarray,
    candidate: np.ndarray,
    damage_mask: np.ndarray,
    gt_object_mask: np.ndarray,
    eval_cfg: EvalCfg,
    seg_cfg: SegmentCfg,
) -> dict[str, float]:
    """All §8.1 metrics for one candidate (LPIPS: lower better, others: higher better)."""
    return {
        "lpips": lpips_score(gt, candidate, damage_mask, eval_cfg),
        "psnr": masked_psnr(gt, candidate, damage_mask),
        "ssim": masked_ssim(gt, candidate, damage_mask),
        "delta_e_boundary": boundary_delta_e(gt, candidate, damage_mask, eval_cfg.boundary_ring_px),
        "silhouette_iou": silhouette_iou(candidate, gt_object_mask, damage_mask, seg_cfg),
    }
