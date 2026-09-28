"""IO round-trip and atomicity tests; ground-truth access guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from counterpart import io as cio
from counterpart.types import Sample


def test_image_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cio.write_image(path, img)
    back = cio.read_image(path)
    assert back.shape == img.shape
    assert np.array_equal(back, img)


def test_mask_round_trip(tmp_path: Path) -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 3:12] = True
    path = tmp_path / "mask.png"
    cio.write_mask(path, mask)
    assert np.array_equal(cio.read_mask(path), mask)


def test_atomic_writes_leave_no_junk(tmp_path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cio.write_image(tmp_path / "a.png", img)
    cio.save_json(tmp_path / "b.json", {"x": 1})
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert leftovers == [], f"temporary files left behind: {leftovers}"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rec.jsonl"
    cio.append_jsonl(path, [{"idx": 0}, {"idx": 1}])
    cio.append_jsonl(path, [{"idx": 2}])
    assert [r["idx"] for r in cio.read_jsonl(path)] == [0, 1, 2]
    assert cio.read_jsonl(tmp_path / "missing.jsonl") == []


def test_sample_gt_hidden_guard(tmp_path: Path) -> None:
    sample = Sample(sample_id="s0", split="test", root=tmp_path, gt_hidden=True)
    with pytest.raises(PermissionError):
        _ = sample.original_path
    visible = Sample(sample_id="s0", split="test", root=tmp_path)
    assert visible.original_path == tmp_path / "original.png"
