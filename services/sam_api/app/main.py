import io
import json
import logging
import time
import zipfile
from datetime import datetime, timezone

from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile

from .constants import PRELOAD_MODELS
from .contracts import SegmentMetadata, SegmentMetadataItem
from .models import load_resources
from .service import segment_with_sam


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="SAM Segmentation API")
app.state.ready = False


@app.on_event("startup")
async def preload_models() -> None:
    if not PRELOAD_MODELS:
        logger.info("PRELOAD_MODELS is disabled; readiness will stay false until first request.")
        logger.info("SAM startup finished at %s (UTC)", datetime.now(timezone.utc).isoformat())
        return
    load_resources()
    app.state.ready = True
    logger.info("SAM startup finished at %s (UTC)", datetime.now(timezone.utc).isoformat())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    if app.state.ready:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model resources are not loaded yet.")


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    bboxes: str = Form(...),
) -> Response:
    started_at = datetime.now(timezone.utc)
    start_monotonic = time.perf_counter()
    logger.info("SAM segment started at %s", started_at.isoformat())

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        parsed_bboxes = json.loads(bboxes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="`bboxes` must be valid JSON.") from exc

    if not isinstance(parsed_bboxes, list):
        raise HTTPException(status_code=400, detail="`bboxes` must be a list.")

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - input parsing guard
        raise HTTPException(status_code=400, detail="Failed to decode image.") from exc

    for bbox in parsed_bboxes:
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise HTTPException(status_code=400, detail="Each bbox must be a list of 4 numbers.")
        if not all(isinstance(value, (int, float)) for value in bbox):
            raise HTTPException(status_code=400, detail="Each bbox must contain only numbers.")

    try:
        resources = load_resources()
        app.state.ready = True
        segmentations = segment_with_sam(image, parsed_bboxes, resources)
    except Exception as exc:
        finished_at = datetime.now(timezone.utc)
        elapsed_ms = (time.perf_counter() - start_monotonic) * 1000
        logger.exception(
            "SAM segment finished at %s (failed, %.1f ms)",
            finished_at.isoformat(),
            elapsed_ms,
        )
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc

    zip_buffer = io.BytesIO()
    metadata_items: list[SegmentMetadataItem] = []
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, segmentation in enumerate(segmentations):
            if segmentation.mask.ndim != 2:
                continue

            mask_image = Image.fromarray((segmentation.mask > 0).astype("uint8") * 255, mode="L")
            png_buffer = io.BytesIO()
            mask_image.save(png_buffer, format="PNG")
            png_bytes = png_buffer.getvalue()
            mask_filename = f"mask_{idx:03d}.png"
            zf.writestr(mask_filename, png_bytes)
            metadata_items.append(
                SegmentMetadataItem(
                    detection_index=idx,
                    mask_filename=mask_filename,
                    mask_score=segmentation.mask_score,
                )
            )

        metadata = SegmentMetadata(instances=metadata_items)
        zf.writestr("metadata.json", metadata.model_dump_json(indent=2))

    finished_at = datetime.now(timezone.utc)
    elapsed_ms = (time.perf_counter() - start_monotonic) * 1000
    logger.info(
        "SAM segment finished at %s (success, %.1f ms, masks=%d)",
        finished_at.isoformat(),
        elapsed_ms,
        len(metadata_items),
    )
    return Response(content=zip_buffer.getvalue(), media_type="application/zip")
