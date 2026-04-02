import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile

from .constants import (
    FLORENCE_API_URL,
    PREDICT_READINESS_TIMEOUT_SECONDS,
    SAM_API_URL,
)
from .contracts import PredictResponse
from .service import run_predict


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Predict Orchestrator API")


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Predict startup finished at %s (UTC)", datetime.now(timezone.utc).isoformat())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _check_downstream_health(url: str) -> None:
    health_url = f"{url}/health"
    request = urllib.request.Request(url=health_url, method="GET")
    with urllib.request.urlopen(request, timeout=PREDICT_READINESS_TIMEOUT_SECONDS):
        return


@app.get("/ready")
async def ready() -> dict[str, str]:
    try:
        _check_downstream_health(FLORENCE_API_URL)
        _check_downstream_health(SAM_API_URL)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"Downstream healthcheck failed: {exc}") from exc
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    started_at = datetime.now(timezone.utc)
    start_monotonic = time.perf_counter()
    logger.info("Predict request started at %s", started_at.isoformat())

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        result = await run_predict(
            file_bytes=file_bytes,
            filename=file.filename or "image.jpg",
            content_type=file.content_type,
        )
    except Exception as exc:
        finished_at = datetime.now(timezone.utc)
        elapsed_ms = (time.perf_counter() - start_monotonic) * 1000
        logger.info(
            "Predict request finished at %s (failed, %.1f ms)",
            finished_at.isoformat(),
            elapsed_ms,
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    finished_at = datetime.now(timezone.utc)
    elapsed_ms = (time.perf_counter() - start_monotonic) * 1000
    logger.info(
        "Predict request finished at %s (success, %.1f ms)",
        finished_at.isoformat(),
        elapsed_ms,
    )
    return result
