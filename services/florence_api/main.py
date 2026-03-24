import io
import logging
import os
import time
from datetime import datetime, timezone

from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile

from contracts import DetectionResponse
from models import load_resources
from service import detect_with_florence


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Florence Detection API")
app.state.ready = False


def _preload_enabled() -> bool:
    return os.getenv("PRELOAD_MODELS", "true").strip().lower() in {"1", "true", "yes", "on"}


@app.on_event("startup")
async def preload_models() -> None:
    if not _preload_enabled():
        logger.info("PRELOAD_MODELS is disabled; readiness will stay false until first request.")
        return
    load_resources()
    app.state.ready = True


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    if app.state.ready:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model resources are not loaded yet.")


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    started_at = datetime.now(timezone.utc)
    start_monotonic = time.perf_counter()
    logger.info("Florence detect started at %s", started_at.isoformat())

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - input parsing guard
        raise HTTPException(status_code=400, detail="Failed to decode image.") from exc

    try:
        resources = load_resources()
        app.state.ready = True
        detections = detect_with_florence(image, resources)
    except Exception as exc:
        finished_at = datetime.now(timezone.utc)
        elapsed_ms = (time.perf_counter() - start_monotonic) * 1000
        logger.exception(
            "Florence detect finished at %s (failed, %.1f ms)",
            finished_at.isoformat(),
            elapsed_ms,
        )
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    finished_at = datetime.now(timezone.utc)
    elapsed_ms = (time.perf_counter() - start_monotonic) * 1000
    logger.info(
        "Florence detect finished at %s (success, %.1f ms, detections=%d)",
        finished_at.isoformat(),
        elapsed_ms,
        len(detections),
    )
    return DetectionResponse(detections=detections)

