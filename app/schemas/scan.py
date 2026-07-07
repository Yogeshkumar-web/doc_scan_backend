from pydantic import BaseModel, Field

class ScanResponse(BaseModel):
    success: bool = True
    image_base64: str
    width: int
    height: int
    edge_detected: bool
    confidence: float = 0.0
    crop_confidence: float = 0.0
    crop_method: str = "unknown"
    background_whiteness: float = 0.0
    shadow_score: float = 0.0
    text_contrast: float = 0.0
    blur_score: float = 0.0
    glare_score: float = 0.0
    selected_enhancement: str = "unknown"
    processing_mode: str = "auto"
    warnings: list[str] = Field(default_factory=list)

class ScanErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
