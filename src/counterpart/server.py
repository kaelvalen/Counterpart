"""FastAPI web app for the counterpart demo (SPEC.md §2, §5.5).

Upload an image (and optionally paint the damage mask), optionally give a prompt,
and watch the pipeline work: candidate generation with live progress, classical
scoring, the best completion and the pixel-uncertainty map.

GPU work is serialised through a single-worker job queue; every job also caches its
candidates under ``runs/webapp/jobs/<job_id>/`` so results can be revisited.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from counterpart import io as cio
from counterpart.config import Cfg, load_config
from counterpart.localize.refine import missing_piece_mask, should_be_object_mask
from counterpart.segment.threshold import object_mask, square_crop_box
from counterpart.viz.panels import side_by_side  # noqa: F401  (kept for future panels)
from counterpart.webapp import reconstruct

STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_CONFIG = "configs/default.yaml"

STAGE_LABELS = {
    "prepare": "Görüntü hazırlanıyor",
    "generate": "Adaylar üretiliyor",
    "score": "Klasik terimlerle puanlanıyor",
    "uncertainty": "Belirsizlik haritası",
    "done": "Bitti",
}


def _decode_image(payload: bytes) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("görüntü çözülemedi (desteklenen bir format yükleyin)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_mask(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    gray = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("maske çözülemedi")
    if gray.shape != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return gray > 127


class JobRunner:
    """Single-worker GPU queue with in-memory job state."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------------ public

    def submit(
        self,
        *,
        image_bytes: bytes,
        mask_bytes: bytes | None,
        prompt: str,
        n: int,
        config_path: str,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "error": None,
                "result": None,
            }
        self.executor.submit(self._run, job_id, image_bytes, mask_bytes, prompt, n, config_path)
        return job_id

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return dict(job)

    # ----------------------------------------------------------------- internal

    def _update(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(fields)

    def _run(
        self,
        job_id: str,
        image_bytes: bytes,
        mask_bytes: bytes | None,
        prompt: str,
        n: int,
        config_path: str,
    ) -> None:
        try:
            cfg: Cfg = load_config(config_path)
            self._update(job_id, status="running", stage="prepare", progress=0.02)

            image = _decode_image(image_bytes)
            hand_mask = None
            if mask_bytes:
                candidate = _decode_mask(mask_bytes, image.shape[:2])
                if candidate.any():
                    hand_mask = candidate

            if hand_mask is not None:
                mask = hand_mask
                obj_override = None
                box = mask | object_mask(image, cfg.segment)
            else:
                mask = missing_piece_mask(image, cfg.segment)
                obj_override = should_be_object_mask(image, cfg.segment)
                box = obj_override

            x0, y0, side = square_crop_box(box, cfg.segment.crop_margin)
            image = image[y0 : y0 + side, x0 : x0 + side]
            mask = mask[y0 : y0 + side, x0 : x0 + side]
            if obj_override is not None:
                obj_override = obj_override[y0 : y0 + side, x0 : x0 + side]

            result = reconstruct(
                cfg,
                image,
                mask,
                n=n,
                prompt=prompt or None,
                object_mask_override=obj_override,
                progress=lambda stage, fraction: self._update(
                    job_id, stage=stage, progress=round(fraction, 3)
                ),
            )

            job_dir = Path(cfg.project.paths.runs_dir) / "webapp" / "jobs" / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            cio.write_image(job_dir / "damaged.png", result["damaged"])  # type: ignore[arg-type]
            cio.write_mask(job_dir / "mask.png", result["damage_mask"])  # type: ignore[arg-type]
            cio.write_image(job_dir / "best.png", result["best"])  # type: ignore[arg-type]
            cio.write_image(job_dir / "uncertainty.png", result["uncertainty"])  # type: ignore[arg-type]
            cio.write_image(job_dir / "candidates.png", result["candidates"])  # type: ignore[arg-type]

            gallery: list[dict[str, Any]] = []
            for idx, image_candidate, score in zip(
                result["candidate_idx"],  # type: ignore[arg-type]
                result["candidate_images"],  # type: ignore[arg-type]
                result["combined_scores"],  # type: ignore[arg-type]
                strict=True,
            ):
                name = f"cand_{int(idx):03d}.png"
                cio.write_image(job_dir / name, image_candidate)
                gallery.append(
                    {
                        "idx": int(idx),
                        "score": round(float(score), 3),
                        "url": f"/api/jobs/{job_id}/files/{name}",
                        "best": int(idx) == int(result["best_idx"]),
                    }
                )
            gallery.sort(key=lambda item: -item["score"])

            self._update(
                job_id,
                status="done",
                stage="done",
                progress=1.0,
                result={
                    "images": {
                        "damaged": f"/api/jobs/{job_id}/files/damaged.png",
                        "mask": f"/api/jobs/{job_id}/files/mask.png",
                        "best": f"/api/jobs/{job_id}/files/best.png",
                        "uncertainty": f"/api/jobs/{job_id}/files/uncertainty.png",
                        "candidates_strip": f"/api/jobs/{job_id}/files/candidates.png",
                    },
                    "candidates": gallery,
                    "best_idx": int(result["best_idx"]),
                    "best_score": round(float(result["best_score"]), 3),
                    "mean_uncertainty": round(float(result["mean_uncertainty"]), 3),
                    "n": n,
                    "hand_mask": hand_mask is not None,
                },
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self._update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


RUNNER = JobRunner()
app = FastAPI(title="counterpart", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    import torch

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "stages": STAGE_LABELS,
    }


@app.post("/api/jobs")
async def create_job(
    image: Annotated[UploadFile, File()],
    mask: Annotated[UploadFile | None, File()] = None,
    prompt: Annotated[str, Form()] = "",
    n: Annotated[int, Form()] = 8,
) -> dict[str, str]:
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="boş görüntü")
    mask_bytes = await mask.read() if mask is not None else None
    if mask_bytes is not None and not mask_bytes:
        mask_bytes = None
    n = max(1, min(int(n), 24))
    job_id = RUNNER.submit(
        image_bytes=payload,
        mask_bytes=mask_bytes,
        prompt=prompt.strip(),
        n=n,
        config_path=DEFAULT_CONFIG,
    )
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    try:
        job = RUNNER.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="iş bulunamadı") from exc
    job["stage_label"] = STAGE_LABELS.get(str(job.get("stage")), job.get("stage"))
    return job


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str) -> FileResponse:
    safe = Path(name).name
    if safe != name or not safe.endswith(".png"):
        raise HTTPException(status_code=400, detail="geçersiz dosya adı")
    runs_dir = Path(load_config(DEFAULT_CONFIG).project.paths.runs_dir)
    path = runs_dir / "webapp" / "jobs" / job_id / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="dosya yok")
    return FileResponse(path, media_type="image/png")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the app with uvicorn (imported lazily so tests do not need it)."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
