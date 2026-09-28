#!/usr/bin/env python
"""Per-term CPU cost measurement (Faz 3 acceptance: <200 ms per candidate target).

For N representative samples: build the score context once, prepare each term, then
time ``score`` over the sample's candidates. Reports mean/p95 per candidate.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from counterpart import io as cio
from counterpart.config import load_config

# register terms
from counterpart.score import (  # noqa: F401
    boundary,
    contour,
    frequency,
    rotational,
    symmetry,
    texture,
)
from counterpart.score.base import TERMS, build_score_sample
from counterpart.score.combine import DEFAULT_TERMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--split", default="gonogo")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--terms", default=",".join(DEFAULT_TERMS))
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs_dir = Path(cfg.project.paths.runs_dir)
    console = Console()

    sample_ids = [
        sid
        for sid in cio.iter_sample_ids(runs_dir, args.experiment, args.split)
        if cio.read_jsonl(runs_dir / args.experiment / args.split / sid / "candidates.jsonl")
    ][: args.samples]
    if not sample_ids:
        raise SystemExit(f"no scored-ready samples under {runs_dir / args.experiment / args.split}")

    term_names = [t.strip() for t in args.terms.split(",")]
    timings: dict[str, dict[str, list[float]]] = {
        name: {"prepare": [], "score": []} for name in term_names
    }

    for sample_id in sample_ids:
        sample = cio.load_sample(runs_dir, args.experiment, args.split, sample_id, gt_hidden=True)
        ctx = build_score_sample(sample, cfg)
        records = cio.read_jsonl(sample.candidates_manifest_path)[: args.candidates]
        candidates = [cio.read_image(sample.candidate_path(int(rec["idx"]))) for rec in records]
        for name in term_names:
            term = TERMS[name]()
            t0 = time.perf_counter()
            prepared = term.prepare(ctx)
            timings[name]["prepare"].append((time.perf_counter() - t0) * 1000.0)
            for candidate in candidates:
                t0 = time.perf_counter()
                term.score(prepared, candidate)
                timings[name]["score"].append((time.perf_counter() - t0) * 1000.0)

    table = Table(
        title=f"score term timings — {len(sample_ids)} samples, {args.candidates} candidates"
    )
    for column in ("term", "prepare ms", "score mean ms", "score p95 ms", "candidates/s"):
        table.add_column(column)
    for name in term_names:
        score_values = timings[name]["score"]
        mean_ms = statistics.mean(score_values)
        p95_ms = float(np.percentile(score_values, 95))
        table.add_row(
            name,
            f"{statistics.mean(timings[name]['prepare']):.1f}",
            f"{mean_ms:.1f}",
            f"{p95_ms:.1f}",
            f"{1000.0 / mean_ms:.1f}",
        )
    console.print(table)


if __name__ == "__main__":
    main()
