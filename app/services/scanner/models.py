from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    points: np.ndarray | None
    confidence: float
    method: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CropResult:
    image: np.ndarray
    edge_detected: bool
    confidence: float
    method: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QualityMetrics:
    background_whiteness: float
    shadow_score: float
    text_contrast: float
    blur_score: float = 0.0
    glare_score: float = 0.0
    binary_ink_ratio: float = 0.0
    selected_enhancement: str = "unknown"


@dataclass(frozen=True)
class EnhancementResult:
    image: np.ndarray
    metrics: QualityMetrics
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineResult:
    image: np.ndarray
    edge_detected: bool
    crop_confidence: float
    crop_method: str
    metrics: QualityMetrics
    warnings: list[str] = field(default_factory=list)
