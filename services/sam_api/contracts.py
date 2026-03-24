from pydantic import BaseModel


class SegmentMetadataItem(BaseModel):
    detection_index: int
    mask_filename: str
    mask_score: float | None = None
    mask_area: int


class SegmentMetadata(BaseModel):
    instances: list[SegmentMetadataItem]

