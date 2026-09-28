"""Core runtime types: Sample, Candidate, CandidateSpec, TermResult.

Conventions (project-wide, see SPEC.md §13):
  * images: ``np.uint8`` ``HxWx3`` **RGB** (OpenCV BGR conversion lives in ``io.py`` only)
  * masks:  ``np.bool_`` ``HxW`` (True = region of interest); on disk 0/255 PNG
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Sample:
    """One processed sample; ``root`` is its cache directory under ``runs/``.

    ``gt_hidden`` is set when the sample is handed to code that must not see
    ground truth (scoring stage). Accessing :attr:`original_path` then raises.
    """

    sample_id: str
    split: str
    root: Path
    meta: dict[str, Any] = field(default_factory=dict)
    gt_hidden: bool = False

    @property
    def original_path(self) -> Path:
        if self.gt_hidden:
            raise PermissionError(
                "ground truth (original.png) is not accessible in this phase; "
                "score/ code must never read it (SPEC.md §13)"
            )
        return self.root / "original.png"

    @property
    def damaged_path(self) -> Path:
        return self.root / "damaged.png"

    @property
    def damage_mask_path(self) -> Path:
        return self.root / "damage_mask.png"

    @property
    def gen_mask_path(self) -> Path:
        return self.root / "gen_mask.png"

    @property
    def object_mask_path(self) -> Path:
        return self.root / "object_mask.png"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def candidates_manifest_path(self) -> Path:
        return self.root / "candidates.jsonl"

    @property
    def scores_path(self) -> Path:
        return self.root / "scores.parquet"

    @property
    def baseline_scores_path(self) -> Path:
        return self.root / "baseline_scores.parquet"

    @property
    def selection_path(self) -> Path:
        return self.root / "selection.json"

    @property
    def modes_path(self) -> Path:
        return self.root / "modes.json"

    @property
    def uncertainty_path(self) -> Path:
        return self.root / "uncertainty.npy"

    @property
    def panel_path(self) -> Path:
        return self.root / "panel.png"

    def candidate_path(self, idx: int) -> Path:
        return self.candidates_dir / f"cand_{idx:03d}.png"


@dataclass(slots=True)
class CandidateSpec:
    """Generation parameters for one candidate, decided before the GPU call."""

    idx: int
    seed: int
    prompt: str
    guidance_scale: float
    num_inference_steps: int


@dataclass(slots=True)
class Candidate:
    """A generated candidate and the parameters that produced it."""

    idx: int
    path: Path
    seed: int
    prompt: str
    guidance_scale: float
    num_inference_steps: int
    generator_id: str
    seconds: float | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> Candidate:
        record = dict(record)
        record["path"] = Path(record["path"])
        return cls(**record)


@dataclass(slots=True)
class TermResult:
    """One score term's output for one candidate. Higher value = better."""

    value: float
    applicable: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)
