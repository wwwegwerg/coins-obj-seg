from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    score: float


class DetectionResponse(BaseModel):
    detections: list[Detection]
