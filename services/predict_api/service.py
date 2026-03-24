import base64
import io
import json
import logging
import zipfile

import httpx
from PIL import Image

from contracts import (
    Detection,
    DetectionResponse,
    InstancePrediction,
    PredictResponse,
    SegmentMetadata,
)
from settings import (
    FLORENCE_API_URL,
    PREDICT_HTTP_TIMEOUT_SECONDS,
    SAM_API_URL,
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


def _mask_bbox_iou(mask_png_bytes: bytes, bbox: list[float]) -> float | None:
    try:
        image = Image.open(io.BytesIO(mask_png_bytes)).convert("RGBA")
    except Exception:
        return None

    alpha = image.getchannel("A")
    alpha_data = alpha.getdata()
    width, height = image.size

    xs: list[int] = []
    ys: list[int] = []
    for idx, value in enumerate(alpha_data):
        if value > 0:
            x = idx % width
            y = idx // width
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return 0.0

    mx1, my1, mx2, my2 = min(xs), min(ys), max(xs), max(ys)
    bx1, by1, bx2, by2 = bbox

    inter_x1 = max(float(mx1), float(bx1))
    inter_y1 = max(float(my1), float(by1))
    inter_x2 = min(float(mx2), float(bx2))
    inter_y2 = min(float(my2), float(by2))

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    bbox_area = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
    mask_bbox_area = max(0.0, float(mx2 - mx1)) * max(0.0, float(my2 - my1))
    union = bbox_area + mask_bbox_area - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


async def run_predict(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> PredictResponse:
    detections = await call_florence(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
    )
    if not detections:
        return PredictResponse(objects=[], instances=[])

    bboxes = [item.bbox for item in detections]
    mask_files, metadata = await call_sam(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
        bboxes=bboxes,
    )

    instances: list[InstancePrediction] = []
    for item in metadata.instances:
        if item.detection_index >= len(detections):
            continue
        detection = detections[item.detection_index]
        mask_bytes = mask_files.get(item.mask_filename)
        if not mask_bytes:
            continue
        instances.append(
            InstancePrediction(
                label=detection.label,
                mask_score=item.mask_score,
                bbox=detection.bbox,
                mask_area=item.mask_area,
                bbox_mask_iou=_mask_bbox_iou(mask_bytes, detection.bbox),
                png_base64=base64.b64encode(mask_bytes).decode("ascii"),
            )
        )

    objects = sorted({item.label for item in instances})
    logger.info(
        "Predict complete: objects=%d instances=%d",
        len(objects),
        len(instances),
    )
    return PredictResponse(objects=objects, instances=instances)

