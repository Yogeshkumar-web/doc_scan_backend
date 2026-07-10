import cv2
import numpy as np

from app.config import settings
from app.services.scanner.detection import crop_document
from app.services.scanner.enhancement import enhance_document, enhance_preselected_auto_candidate
from app.services.scanner.geometry import four_point_transform
from app.services.scanner.models import PipelineResult, QualityMetrics
from app.services.scanner.quality import compute_quality_metrics


def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return resize_for_processing(image)


def resize_for_processing(image: np.ndarray) -> np.ndarray:
    max_dimension = max(800, int(settings.scan_max_dimension))
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dimension:
        return image

    scale = max_dimension / max_side
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def build_pipeline_result(
    crop_image: np.ndarray,
    mode: str,
    *,
    edge_detected: bool,
    crop_confidence: float,
    crop_method: str,
    crop_warnings: list[str] | None = None,
    selected_enhancement: str | None = None,
) -> PipelineResult:
    capture_metrics = compute_quality_metrics(crop_image)
    if mode == "auto" and selected_enhancement:
        enhancement = enhance_preselected_auto_candidate(crop_image, selected_enhancement)
    else:
        enhancement = enhance_document(crop_image, mode)
    metrics = QualityMetrics(
        background_whiteness=enhancement.metrics.background_whiteness,
        shadow_score=enhancement.metrics.shadow_score,
        text_contrast=enhancement.metrics.text_contrast,
        blur_score=capture_metrics.blur_score,
        glare_score=capture_metrics.glare_score,
        binary_ink_ratio=enhancement.metrics.binary_ink_ratio,
        selected_enhancement=enhancement.metrics.selected_enhancement,
    )
    warnings = [
        *(crop_warnings or []),
        *(w for w in enhancement.warnings if w not in {"IMAGE_BLURRY", "GLARE_DETECTED"}),
    ]
    if metrics.blur_score > 0.72:
        warnings.append("IMAGE_BLURRY")
    if metrics.glare_score > 0.08:
        warnings.append("GLARE_DETECTED")

    return PipelineResult(
        image=enhancement.image,
        edge_detected=edge_detected,
        crop_confidence=crop_confidence,
        crop_method=crop_method,
        metrics=metrics,
        warnings=sorted(set(warnings), key=warnings.index),
    )


def process_document(image_bytes: bytes, mode: str = "auto") -> PipelineResult:
    image = decode_image(image_bytes)

    crop = crop_document(image)
    return build_pipeline_result(
        crop.image,
        mode,
        edge_detected=crop.edge_detected,
        crop_confidence=crop.confidence,
        crop_method=crop.method,
        crop_warnings=crop.warnings,
    )


def process_full_document(image_bytes: bytes, mode: str = "auto") -> PipelineResult:
    image = decode_image(image_bytes)
    return build_pipeline_result(
        image,
        mode,
        edge_detected=False,
        crop_confidence=1.0,
        crop_method="full_image_confirmed",
        crop_warnings=[],
    )


def process_document_with_corners(
    image_bytes: bytes,
    points: list[dict[str, float]],
    mode: str = "auto",
    selected_enhancement: str | None = None,
) -> PipelineResult:
    image = decode_image(image_bytes)
    h, w = image.shape[:2]
    if len(points) != 4:
        raise ValueError("Exactly four crop points are required")

    pixel_points = []
    for point in points:
        x = float(point["x"])
        y = float(point["y"])
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("Crop points must be normalized between 0 and 1")
        pixel_points.append([x * (w - 1), y * (h - 1)])

    pts = np.array(pixel_points, dtype="float32")
    warped = four_point_transform(image, pts)
    if warped.shape[0] < 40 or warped.shape[1] < 40:
        raise ValueError("Selected crop area is too small")

    return build_pipeline_result(
        warped,
        mode,
        edge_detected=True,
        crop_confidence=1.0,
        crop_method="manual_corners",
        crop_warnings=[],
        selected_enhancement=selected_enhancement,
    )
