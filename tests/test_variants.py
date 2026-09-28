"""Candidate planning tests: counts, determinism, combo distribution, prompt filling."""

from __future__ import annotations

import pytest

from counterpart.config import GeneratorCfg
from counterpart.generate.variants import plan_candidates, stable_seed


def test_stable_seed_is_deterministic() -> None:
    assert stable_seed("sample-a", 3) == stable_seed("sample-a", 3)
    assert stable_seed("sample-a", 3) != stable_seed("sample-b", 3)
    assert stable_seed("sample-a", 3) != stable_seed("sample-a", 4)


def test_plan_count_and_determinism() -> None:
    cfg = GeneratorCfg()
    a = plan_candidates(32, cfg, "s0", category="vase")
    b = plan_candidates(32, cfg, "s0", category="vase")
    assert a == b, "same sample id must give identical schedules"
    assert [s.idx for s in a] == list(range(32))
    seeds = [s.seed for s in a]
    assert len(set(seeds)) == len(seeds), "seeds must be unique within a sample"


def test_plan_covers_all_combos() -> None:
    cfg = GeneratorCfg()
    specs = plan_candidates(len(cfg.prompts) * len(cfg.guidance_scales), cfg, "s1", category="mug")
    combos = {(s.prompt, s.guidance_scale) for s in specs}
    assert len(combos) == len(cfg.prompts) * len(cfg.guidance_scales)


def test_prompt_category_filling() -> None:
    cfg = GeneratorCfg(prompts=["a complete intact {category}"])
    spec = plan_candidates(1, cfg, "s2", category="lamp")[0]
    assert spec.prompt == "a complete intact lamp"
    spec_default = plan_candidates(1, cfg, "s2")[0]
    assert spec_default.prompt == "a complete intact object"


def test_plan_rejects_bad_n() -> None:
    with pytest.raises(ValueError):
        plan_candidates(0, GeneratorCfg(), "s3")
