from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    score: float


class DetectionResponse(BaseModel):
    detections: list[Detection]


class SegmentMetadataItem(BaseModel):
    detection_index: int
    mask_filename: str
    mask_score: float | None = None


class SegmentMetadata(BaseModel):
    instances: list[SegmentMetadataItem]


class InstancePrediction(BaseModel):
    label: str
    mask_score: float | None = None
    bbox: list[float] = Field(min_length=4, max_length=4)
    bbox_mask_iou: float | None = None
    png_base64: str


class PredictResponse(BaseModel):
    objects: list[str]
    instances: list[InstancePrediction]

