import base64
import io
import json
import logging
import zipfile
from typing import Iterable

import httpx
import numpy as np
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

# Hardcoded detection and instance post-processing parameters.
CONFIDENCE_THRESHOLD = 0.0
MIN_BOX_WIDTH_PX = 4.0
MIN_BOX_HEIGHT_PX = 4.0
MIN_BOX_AREA_RATIO = 0.001
MAX_BOX_AREA_RATIO = 0.95
MIN_BOX_ASPECT_RATIO = 0.2
MAX_BOX_ASPECT_RATIO = 5.0
MIN_CUTOUT_WIDTH_PX = 32
MIN_CUTOUT_HEIGHT_PX = 32
MIN_MASK_SCORE = 0.7
MIN_BBOX_MASK_IOU = 0.3
MASK_NMS_IOU_THRESHOLD = 0.88


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


def _clamp_bbox(bbox: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]


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


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _passes_cutout_size(mask: np.ndarray) -> bool:
    bounds = _mask_bounds(mask)
    if bounds is None:
        return False
    x_min, y_min, x_max, y_max = bounds
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return width >= MIN_CUTOUT_WIDTH_PX and height >= MIN_CUTOUT_HEIGHT_PX


def _mask_bbox_iou(mask: np.ndarray, bbox: list[float]) -> float | None:
    bounds = _mask_bounds(mask)
    if bounds is None:
        return 0.0

    mx1, my1, mx2, my2 = bounds
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


def _passes_box_sanity_checks(bbox: list[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0.0 or box_h <= 0.0:
        return False
    if box_w < MIN_BOX_WIDTH_PX or box_h < MIN_BOX_HEIGHT_PX:
        return False

    image_area = float(width * height)
    if image_area <= 0.0:
        return False
    area_ratio = (box_w * box_h) / image_area
    if area_ratio < MIN_BOX_AREA_RATIO or area_ratio > MAX_BOX_AREA_RATIO:
        return False

    aspect_ratio = box_w / box_h
    if aspect_ratio < MIN_BOX_ASPECT_RATIO or aspect_ratio > MAX_BOX_ASPECT_RATIO:
        return False
    return True


def _mask_iou(lhs_mask: np.ndarray, rhs_mask: np.ndarray) -> float:
    if lhs_mask.shape != rhs_mask.shape:
        return 0.0
    lhs = lhs_mask > 0
    rhs = rhs_mask > 0
    union = np.logical_or(lhs, rhs).sum(dtype=np.int64)
    if union <= 0:
        return 0.0
    intersection = np.logical_and(lhs, rhs).sum(dtype=np.int64)
    return float(intersection) / float(union)


def _apply_mask_nms(
    candidates: list[tuple[InstancePrediction, np.ndarray, float]],
) -> list[InstancePrediction]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item[0].mask_score or 0.0,
            item[2],
            item[0].bbox_mask_iou or 0.0,
        ),
        reverse=True,
    )

    selected_instances: list[InstancePrediction] = []
    selected_masks: list[np.ndarray] = []
    for instance, mask, _ in ordered:
        is_duplicate = any(
            _mask_iou(mask, selected_mask) >= MASK_NMS_IOU_THRESHOLD
            for selected_mask in selected_masks
        )
        if is_duplicate:
            continue
        selected_instances.append(instance)
        selected_masks.append(mask)
    return selected_instances


def _post_process_detections(image: Image.Image, detections: list[Detection]) -> list[Detection]:
    prefiltered: list[Detection] = []
    for item in detections:
        if item.score < CONFIDENCE_THRESHOLD:
            continue
        clamped_bbox = _clamp_bbox(item.bbox, image.width, image.height)
        if not _passes_box_sanity_checks(clamped_bbox, image.width, image.height):
            continue
        prefiltered.append(item.model_copy(update={"bbox": clamped_bbox}))

    prefiltered.sort(key=lambda item: item.score, reverse=True)
    return prefiltered


async def run_predict(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> PredictResponse:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image in predict service: {exc}") from exc

    detections = await call_florence(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
    )
    if not detections:
        return PredictResponse(objects=[], instances=[])

    detections = _post_process_detections(image, detections)
    if not detections:
        return PredictResponse(objects=[], instances=[])

    bboxes = [item.bbox for item in detections]
    mask_files, metadata = await call_sam(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
        bboxes=bboxes,
    )

    candidates: list[tuple[InstancePrediction, np.ndarray, float]] = []
    for item in metadata.instances:
        if item.detection_index >= len(detections):
            continue
        detection = detections[item.detection_index]
        clamped_bbox = _clamp_bbox(detection.bbox, image.width, image.height)
        mask_bytes = mask_files.get(item.mask_filename)
        if not mask_bytes:
            continue
        mask = _load_binary_mask(mask_bytes)
        if mask is None:
            continue
        if not _passes_cutout_size(mask):
            continue
        mask_score = item.mask_score
        bbox_mask_iou = _mask_bbox_iou(mask, clamped_bbox)
        if mask_score is None or bbox_mask_iou is None:
            continue
        if mask_score < MIN_MASK_SCORE or bbox_mask_iou < MIN_BBOX_MASK_IOU:
            continue
        cutout_png_bytes = _build_cutout_png_bytes(image, mask)
        encoded_png_bytes = cutout_png_bytes or mask_bytes
        candidates.append(
            (
                InstancePrediction(
                    label=detection.label,
                    mask_score=mask_score,
                    bbox=clamped_bbox,
                    bbox_mask_iou=bbox_mask_iou,
                    png_base64=base64.b64encode(encoded_png_bytes).decode("ascii"),
                ),
                mask,
                detection.score,
            )
        )

    instances = _apply_mask_nms(candidates)
    objects = sorted({item.label for item in instances})
    logger.info(
        "Predict complete: objects=%d instances=%d",
        len(objects),
        len(instances),
    )
    return PredictResponse(objects=objects, instances=instances)

