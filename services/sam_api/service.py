import io
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from models import SamResources


@dataclass
class SegmentationCandidate:
    mask: np.ndarray
    mask_score: float | None = None


def _extract_binary_mask(mask_tensor: Any, candidate_index: int = 0) -> np.ndarray:
    if isinstance(mask_tensor, list):
        if not mask_tensor:
            return np.zeros((0, 0), dtype=np.uint8)
        candidate_index = max(0, min(candidate_index, len(mask_tensor) - 1))
        mask_tensor = mask_tensor[candidate_index]

    if isinstance(mask_tensor, torch.Tensor):
        tensor = mask_tensor.detach().cpu()
        if tensor.ndim == 3:
            candidate_index = max(0, min(candidate_index, tensor.shape[0] - 1))
            tensor = tensor[candidate_index]
        while tensor.ndim > 2:
            tensor = tensor[0]
        return (tensor.numpy() > 0.0).astype(np.uint8)

    array = np.asarray(mask_tensor)
    if array.ndim == 3:
        candidate_index = max(0, min(candidate_index, array.shape[0] - 1))
        array = array[candidate_index]
    while array.ndim > 2:
        array = array[0]
    return (array > 0.0).astype(np.uint8)


def segment_with_sam(
    image: Image.Image,
    bboxes: list[list[float]],
    resources: SamResources,
) -> list[SegmentationCandidate]:
    if not bboxes:
        return []

    normalized_bboxes = [clamp_bbox(bbox, image.width, image.height) for bbox in bboxes]
    input_boxes = [normalized_bboxes]
    sam_inputs = resources.processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(resources.device)

    with torch.no_grad():
        sam_outputs = resources.model(**sam_inputs)

    post_masks = resources.processor.post_process_masks(
        sam_outputs.pred_masks,
        sam_inputs["original_sizes"],
    )
    image_masks = post_masks[0] if post_masks else []

    raw_scores = getattr(sam_outputs, "iou_scores", None)
    iou_scores: torch.Tensor | None = None
    if isinstance(raw_scores, torch.Tensor):
        iou_scores = raw_scores.detach().cpu()
        if iou_scores.ndim == 3:
            iou_scores = iou_scores[0]

    segmentations: list[SegmentationCandidate] = []
    for idx, mask in enumerate(image_masks):
        best_mask_idx = 0
        best_mask_score: float | None = None

        if isinstance(iou_scores, torch.Tensor):
            if iou_scores.ndim == 2 and idx < iou_scores.shape[0]:
                per_mask_scores = iou_scores[idx]
                best_mask_idx = int(torch.argmax(per_mask_scores).item())
                best_mask_score = float(per_mask_scores[best_mask_idx].item())
            elif iou_scores.ndim == 1 and idx < iou_scores.shape[0]:
                best_mask_score = float(iou_scores[idx].item())

        segmentations.append(
            SegmentationCandidate(
                mask=_extract_binary_mask(mask, candidate_index=best_mask_idx),
                mask_score=best_mask_score,
            )
        )
    return segmentations


def build_cutout_png_bytes(image: Image.Image, mask: np.ndarray) -> tuple[bytes | None, int]:
    if mask.ndim != 2:
        return None, 0

    mask_area = int(mask.sum())
    if mask_area <= 0:
        return None, 0

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None, 0

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    rgba = image.convert("RGBA")
    rgba_array = np.array(rgba, dtype=np.uint8)
    rgba_array[:, :, 3] = (mask * 255).astype(np.uint8)

    cutout = Image.fromarray(rgba_array, mode="RGBA").crop(
        (x_min, y_min, x_max + 1, y_max + 1)
    )

    buf = io.BytesIO()
    cutout.save(buf, format="PNG")
    return buf.getvalue(), mask_area


def clamp_bbox(bbox: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]