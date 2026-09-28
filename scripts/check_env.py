#!/usr/bin/env python
"""Faz 0 environment check (SPEC.md §3.2, §11).

Verifies, in order:
  1. torch + CUDA availability, device name and compute capability (expects (12, 0) Blackwell)
  2. a small fp16 matmul benchmark
  3. diffusers inpainting pipeline download/load (configured model, fp16)
  4. one real inpainting call, then timed calls -> seconds/image + peak VRAM

Writes ``env_report.json`` + ``sample_inpaint.png`` under ``--out`` and exits nonzero
if any hard requirement fails.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from counterpart.config import load_config
from counterpart.generate.sd_inpaint import InpaintGenerator
from counterpart.types import CandidateSpec


def synthetic_sample(size: int) -> tuple[np.ndarray, np.ndarray]:
    """A structured synthetic 'object' with a rectangular damage region."""
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    cv2_img = img  # RGB
    yy, xx = np.mgrid[0:size, 0:size]
    body = ((xx - size / 2) / (size * 0.30)) ** 2 + ((yy - size * 0.55) / (size * 0.36)) ** 2 < 1.0
    cv2_img[body] = (70, 90, 160)
    neck = (np.abs(xx - size / 2) < size * 0.07) & (yy > size * 0.10) & (yy < size * 0.30)
    cv2_img[neck] = (60, 75, 140)
    mask = np.zeros((size, size), dtype=bool)
    mask[int(size * 0.42) : int(size * 0.66), int(size * 0.62) : int(size * 0.92)] = True
    assert mask.any()
    return cv2_img, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--steps", type=int, default=None, help="override inference steps for the timed runs"
    )
    parser.add_argument("--runs", type=int, default=2, help="number of timed generations")
    parser.add_argument("--out", default="runs/check_env")
    args = parser.parse_args()

    console = Console()
    cfg = load_config(args.config, profile=args.profile)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "generator_id": cfg.generator.id,
    }

    # ---------------------------------------------------------------- 1. torch/CUDA
    console.rule("1. torch / CUDA")
    if not torch.cuda.is_available():
        console.print("[red]CUDA not available[/red]")
        return 1
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    cap = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)
    report |= {
        "cuda": torch.version.cuda,
        "gpu_name": name,
        "compute_capability": list(cap),
        "vram_total_gib": round(props.total_memory / 1024**3, 2),
    }
    console.print(
        f"GPU: [bold]{name}[/bold]  cc={cap}  VRAM={props.total_memory / 1024**3:.1f} GiB"
    )

    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_iter = 10
    for _ in range(n_iter):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = n_iter * 2 * 4096**3 / dt / 1e12
    report["matmul_tflops_fp16"] = round(tflops, 1)
    console.print(f"fp16 matmul: [bold]{tflops:.1f} TFLOP/s[/bold]")
    del a, b
    torch.cuda.empty_cache()

    # ------------------------------------------------------------- 2. pipeline load
    console.rule("2. diffusers pipeline")
    t0 = time.perf_counter()
    generator = InpaintGenerator.load(cfg.generator)
    load_seconds = time.perf_counter() - t0
    report["pipeline_load_seconds"] = round(load_seconds, 2)
    drain = torch.cuda.memory_allocated() / 1024**3
    report["vram_model_gib"] = round(drain, 2)
    console.print(
        f"loaded [bold]{cfg.generator.id}[/bold] in {load_seconds:.1f}s  (weights {drain:.2f} GiB)"
    )

    # ----------------------------------------------------------- 3. generation calls
    console.rule("3. generation")
    size = cfg.generator.resolution
    image, damage_mask = synthetic_sample(size)

    warm_spec = CandidateSpec(
        idx=0, seed=1234, prompt="", guidance_scale=7.5, num_inference_steps=8
    )
    _warm_img, warm_seconds = generator.generate_one(image, damage_mask, warm_spec)
    console.print(f"warmup (8 steps): {warm_seconds:.2f}s")

    steps = args.steps or cfg.generator.num_inference_steps
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    last = None
    for i in range(args.runs):
        spec = CandidateSpec(
            idx=i,
            seed=2000 + i,
            prompt="",
            guidance_scale=7.5,
            num_inference_steps=steps,
        )
        last, seconds = generator.generate_one(image, damage_mask, spec)
        times.append(seconds)
        console.print(f"run {i}: {seconds:.2f}s  ({steps} steps)")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    report |= {
        "steps": steps,
        "seconds_per_image": round(float(np.mean(times)), 2),
        "seconds_std": round(float(np.std(times)), 2),
        "peak_vram_gib": round(peak, 2),
    }
    console.print(
        f"throughput: [bold]{np.mean(times):.2f} s/image[/bold] ({steps} steps)  "
        f"peak VRAM {peak:.2f} GiB"
    )

    generator.unload()

    # ------------------------------------------------------------------- 4. outputs
    from PIL import Image as PILImage

    PILImage.fromarray(last).save(out_dir / "sample_inpaint.png")
    (out_dir / "env_report.json").write_text(json.dumps(report, indent=2) + "\n")

    table = Table(title="check_env")
    table.add_column("key")
    table.add_column("value")
    for key, value in report.items():
        table.add_row(str(key), str(value))
    console.print(table)
    console.print(f"saved {out_dir}/env_report.json")
    console.print("[green]check_env: OK[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
