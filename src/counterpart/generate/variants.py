"""Candidate parameter planning (SPEC.md §5.3).

N candidates are spread over (prompt x guidance) combinations in round-robin order;
each candidate gets a deterministic seed derived from the sample id, so a rerun of
the same sample reproduces exactly the same candidate set.
"""

from __future__ import annotations

import hashlib
import itertools

from counterpart.config import GeneratorCfg
from counterpart.types import CandidateSpec


def stable_seed(sample_id: str, idx: int, base_seed: int = 0) -> int:
    """Deterministic 31-bit seed: ``hash(sample_id) + idx + base_seed`` (SPEC.md §5.3)."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    h = int.from_bytes(digest[:4], "little")
    return (base_seed + h + idx) % (2**31 - 1)


def plan_candidates(
    n: int,
    cfg: GeneratorCfg,
    sample_id: str,
    category: str | None = None,
) -> list[CandidateSpec]:
    """Build the generation schedule for one sample.

    Kombinasyonlar (prompt, guidance) çarpımıdır ve sırayla dağıtılır; N kombinasyon
    sayısından büyükse baştan başlanır (seed'ler yine de tekil kalır).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    combo_idx = list(itertools.product(range(len(cfg.prompts)), cfg.guidance_scales))
    if not combo_idx:
        raise ValueError("generator config has no (prompt, guidance) combinations")

    specs: list[CandidateSpec] = []
    for idx in range(n):
        prompt_i, guidance = combo_idx[idx % len(combo_idx)]
        prompt = cfg.prompts[prompt_i].format(category=category or "object")
        specs.append(
            CandidateSpec(
                idx=idx,
                seed=stable_seed(sample_id, idx, base_seed=cfg.seed),
                prompt=prompt,
                guidance_scale=float(guidance),
                num_inference_steps=cfg.num_inference_steps,
            )
        )
    return specs
