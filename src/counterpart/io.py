"""Cache & manifest IO.

* Atomic writes: temp file first, then ``os.replace`` (interrupted runs never leave
  half-written outputs behind).
* All OpenCV BGR conversion is centralized here; the rest of the codebase only sees
  RGB ``np.uint8`` arrays and bool masks (SPEC.md §13).
* Image/mask/candidate/JSON/JSONL helpers used by every stage.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from counterpart.types import Candidate, Sample


def _atomic(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Write via a temp file in the same directory, then rename over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- images


def read_image(path: Path) -> np.ndarray:
    """Read an image as ``uint8 HxWx3 RGB``."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an RGB ``uint8 HxWx3`` image as PNG (atomically)."""
    if image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError(f"expected uint8 HxWx3 image, got {image.dtype} {image.shape}")

    def _w(tmp: Path) -> None:
        if not cv2.imwrite(str(tmp), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"cv2.imwrite failed for {tmp}")

    _atomic(path, _w)


def read_mask(path: Path) -> np.ndarray:
    """Read a 0/255 PNG as a bool mask ``HxW``."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    return gray > 127


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a bool mask as a 0/255 PNG (atomically)."""
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError(f"expected bool HxW mask, got {mask.dtype} {mask.shape}")
    payload = (mask.astype(np.uint8)) * 255

    def _w(tmp: Path) -> None:
        if not cv2.imwrite(str(tmp), payload):
            raise OSError(f"cv2.imwrite failed for {tmp}")

    _atomic(path, _w)


# -------------------------------------------------------------------------------- json


def save_json(path: Path, payload: Any) -> None:
    def _w(tmp: Path) -> None:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    _atomic(path, _w)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------------------ sample


def load_sample(
    runs_dir: Path, experiment: str, split: str, sample_id: str, *, gt_hidden: bool = False
) -> Sample:
    """Load a cached sample; ``gt_hidden=True`` for scoring/selection phases."""
    root = Path(runs_dir) / experiment / split / sample_id
    meta_path = root / "meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    return Sample(sample_id=sample_id, split=split, root=root, meta=meta, gt_hidden=gt_hidden)


def iter_sample_ids(runs_dir: Path, experiment: str, split: str) -> list[str]:
    base = Path(runs_dir) / experiment / split
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


# ------------------------------------------------------------------------- candidates


def candidate_records(sample: Sample) -> list[dict[str, Any]]:
    return read_jsonl(sample.candidates_manifest_path)


def load_candidate(sample: Sample, record: dict[str, Any]) -> Candidate:
    return Candidate.from_json(record)
