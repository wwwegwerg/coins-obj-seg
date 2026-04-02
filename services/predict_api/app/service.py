import base64
import io
import json
import logging
import zipfile

import httpx
import numpy as np
from PIL import Image, ImageOps

from .constants import (
    FLORENCE_API_URL,
    PREDICT_HTTP_TIMEOUT_SECONDS,
    SAM_API_URL,
)
from .contracts import (
    Detection,
    DetectionResponse,
    InstancePrediction,
    PredictResponse,
    SegmentMetadata,
)


logger = logging.getLogger(__name__)


async def _post_multipart(
    url: str,
    files: dict[str, tuple[str, bytes, str]],
    accept: str,
    data: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    timeout = httpx.Timeout(PREDICT_HTTP_TIMEOUT_SECONDS)
    headers = {"Accept": accept}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, data=data, files=files, headers=headers)
            response.raise_for_status()
            return response.content, response.headers.get("Content-Type", "")
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise RuntimeError(f"{url} failed with {exc.response.status_code}: {detail}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc}") from exc


async def call_florence(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> list[Detection]:
    response_body, _ = await _post_multipart(
        url=f"{FLORENCE_API_URL}/detect",
        files={"file": (filename, file_bytes, content_type or "application/octet-stream")},
        accept="application/json",
    )
    payload = json.loads(response_body.decode("utf-8"))
    parsed = DetectionResponse.model_validate(payload)
    return parsed.detections


async def call_sam(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    bboxes: list[list[float]],
) -> tuple[dict[str, bytes], SegmentMetadata]:
    response_body, response_content_type = await _post_multipart(
        url=f"{SAM_API_URL}/segment",
        files={"file": (filename, file_bytes, content_type or "application/octet-stream")},
        accept="application/zip",
        data={"bboxes": json.dumps(bboxes, ensure_ascii=False)},
    )
    if "application/zip" not in response_content_type:
        raise RuntimeError(f"SAM API returned unexpected content type: {response_content_type}")

    with zipfile.ZipFile(io.BytesIO(response_body), "r") as zf:
        metadata_bytes = zf.read("metadata.json")
        metadata = SegmentMetadata.model_validate_json(metadata_bytes.decode("utf-8"))
        mask_files: dict[str, bytes] = {}
        for item in metadata.instances:
            mask_files[item.mask_filename] = zf.read(item.mask_filename)
        return mask_files, metadata


def _load_binary_mask(mask_png_bytes: bytes) -> np.ndarray | None:
    try:
        image = Image.open(io.BytesIO(mask_png_bytes)).convert("L")
    except Exception:
        return None
    return (np.array(image, dtype=np.uint8) > 0).astype(np.uint8)


def _build_cutout_png_bytes(image: Image.Image, mask: np.ndarray) -> bytes | None:
    if mask.ndim != 2:
        return None

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    rgba = image.convert("RGBA")
    rgba_array = np.array(rgba, dtype=np.uint8)
    rgba_array[:, :, 3] = (mask * 255).astype(np.uint8)

    cutout = Image.fromarray(rgba_array, mode="RGBA").crop((x_min, y_min, x_max + 1, y_max + 1))
    buf = io.BytesIO()
    cutout.save(buf, format="PNG")
    return buf.getvalue()


async def run_predict(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> PredictResponse:
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image in predict service: {exc}") from exc

    detections = await call_florence(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
    )
    if not detections:
        return PredictResponse(objects=[], instances=[])

    mask_files, metadata = await call_sam(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
        bboxes=[item.bbox for item in detections],
    )

    instances: list[InstancePrediction] = []
    for item in metadata.instances:
        if item.detection_index >= len(detections):
            continue
        mask_bytes = mask_files.get(item.mask_filename)
        if not mask_bytes:
            continue
        mask = _load_binary_mask(mask_bytes)
        if mask is None:
            continue

        detection = detections[item.detection_index]
        encoded_png_bytes = _build_cutout_png_bytes(image, mask) or mask_bytes
        instances.append(
            InstancePrediction(
                label=detection.label,
                confidence_score=detection.score,
                bbox=detection.bbox,
                png_base64=base64.b64encode(encoded_png_bytes).decode("ascii"),
            )
        )

    objects = sorted({item.label for item in instances})
    logger.info(
        "Predict complete: objects=%d instances=%d",
        len(objects),
        len(instances),
    )
    return PredictResponse(objects=objects, instances=instances)
