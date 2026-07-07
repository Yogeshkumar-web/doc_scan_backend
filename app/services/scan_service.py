from dataclasses import dataclass

import numpy as np

from app.services.scanner.detection import crop_document, find_document_candidate
from app.services.scanner.geometry import four_point_transform, order_points


@dataclass(frozen=True)
class ScanResult:
    image: np.ndarray
    edge_detected: bool
    confidence: float
    warnings: list[str]


def find_document_contour(image: np.ndarray) -> np.ndarray | None:
    """Compatibility wrapper for tests and older imports."""
    candidate = find_document_candidate(image)
    return candidate.points if candidate else None


def process_scan(image_bytes: bytes) -> ScanResult:
    """Compatibility wrapper that crops only; enhancement now lives in scanner pipeline."""
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    crop = crop_document(image)
    return ScanResult(
        image=crop.image,
        edge_detected=crop.edge_detected,
        confidence=crop.confidence,
        warnings=crop.warnings,
    )
