from dataclasses import dataclass

import cv2
import numpy as np

from app.services.scanner.geometry import four_point_transform, order_points
from app.services.scanner.models import CropResult, DetectionResult

MAX_DIMENSION_FOR_DETECTION = 1200


@dataclass(frozen=True)
class ContourCandidate:
    points: np.ndarray
    confidence: float
    method: str


def detect_document(image: np.ndarray) -> DetectionResult:
    candidate = find_document_candidate(image)
    if candidate is None:
        return DetectionResult(
            points=None,
            confidence=0.0,
            method="none",
            warnings=["DOCUMENT_EDGES_NOT_FOUND"],
        )
    warnings = [] if candidate.confidence >= 0.62 else ["LOW_CROP_CONFIDENCE"]
    return DetectionResult(
        points=candidate.points,
        confidence=round(candidate.confidence, 2),
        method=candidate.method,
        warnings=warnings,
    )


def crop_document(image: np.ndarray) -> CropResult:
    detection = detect_document(image)
    if detection.points is not None:
        try:
            return CropResult(
                image=four_point_transform(image, detection.points),
                edge_detected=True,
                confidence=detection.confidence,
                method=detection.method,
                warnings=detection.warnings,
            )
        except Exception:
            fallback = trim_non_paper_border(image)
            return CropResult(
                image=fallback.image,
                edge_detected=False,
                confidence=fallback.confidence,
                method=fallback.method,
                warnings=["PERSPECTIVE_CORRECTION_FAILED", *fallback.warnings],
            )

    fallback = trim_non_paper_border(image)
    return fallback

def trim_non_paper_border(image: np.ndarray) -> CropResult:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return CropResult(image=image, edge_detected=False, confidence=0.0, method="none")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bright_threshold = max(105, int(np.percentile(value, 45)))
    low_sat_mask = cv2.inRange(saturation, 0, 115)
    bright_mask = cv2.inRange(value, bright_threshold, 255)
    paper_mask = cv2.bitwise_and(low_sat_mask, bright_mask)

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 8,
    )
    paper_mask = cv2.bitwise_and(paper_mask, adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return CropResult(image=image, edge_detected=False, confidence=0.0,
                           method="full_image", warnings=["USED_FULL_IMAGE"])

    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / float(h * w)
    if area_ratio < 0.45:
        return CropResult(image=image, edge_detected=False, confidence=0.0,
                           method="full_image", warnings=["USED_FULL_IMAGE"])

    # NEW: rotated rect instead of axis-aligned boundingRect — this is the fix.
    # A rotated crop hugs the paper even when it's not perfectly axis-aligned,
    # instead of over-including background at the corners.
    rotated_rect = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(rotated_rect).astype("float32")

    (rect_w, rect_h) = rotated_rect[1]
    if min(rect_w, rect_h) < 10:  # degenerate box guard
        return CropResult(image=image, edge_detected=False, confidence=0.0,
                           method="full_image", warnings=["USED_FULL_IMAGE"])

    cropped = four_point_transform(image, box_points)

    removed_area_ratio = 1.0 - (cropped.shape[0] * cropped.shape[1]) / float(h * w)
    if removed_area_ratio < 0.03:
        return CropResult(image=image, edge_detected=False, confidence=0.0,
                           method="full_image",
                           warnings=["DOCUMENT_EDGES_NOT_FOUND", "USED_FULL_IMAGE"])

    return CropResult(
        image=cropped,
        edge_detected=False,
        confidence=round(float(min(area_ratio, 0.55)), 2),
        method="paper_mask_rotated_trim",
        warnings=["DOCUMENT_EDGES_NOT_FOUND", "BORDER_TRIM_FALLBACK"],
    )


def find_document_candidate(image: np.ndarray) -> ContourCandidate | None:
    orig_h, orig_w = image.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return None

    scale = min(1.0, MAX_DIMENSION_FOR_DETECTION / max(orig_h, orig_w))
    small_w = int(orig_w * scale)
    small_h = int(orig_h * scale)
    if small_w == 0 or small_h == 0:
        return None

    small = cv2.resize(image, (small_w, small_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_maps = _build_edge_maps(small, gray)
    best: ContourCandidate | None = None

    for method, edge_map in edge_maps:
        contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]
        for contour in contours:
            candidate = _candidate_from_contour(contour, small_w, small_h, scale, method)
            if candidate and (best is None or candidate.confidence > best.confidence):
                best = candidate

    if best and best.confidence >= 0.42:
        return best
    return None


def _build_edge_maps(small: np.ndarray, gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    canny = _clean_edges(cv2.Canny(gray, 60, 180))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    adaptive = _clean_edges(cv2.bitwise_not(adaptive))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    bright_page_mask = cv2.inRange(value, 115, 255)
    low_saturation_mask = cv2.inRange(saturation, 0, 95)
    bright_low_sat = _clean_edges(cv2.bitwise_and(bright_page_mask, low_saturation_mask))

    combined = cv2.bitwise_or(cv2.bitwise_or(canny, adaptive), bright_low_sat)
    combined = _clean_edges(combined)

    return [
        ("combined_contour", combined),
        ("canny_contour", canny),
        ("adaptive_contour", adaptive),
        ("paper_mask_contour", bright_low_sat),
    ]


def _clean_edges(edge_map: np.ndarray) -> np.ndarray:
    border_px = 5
    h, w = edge_map.shape[:2]
    if h > 2 * border_px and w > 2 * border_px:
        edge_map[0:border_px, :] = 0
        edge_map[-border_px:, :] = 0
        edge_map[:, 0:border_px] = 0
        edge_map[:, -border_px:] = 0

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return cv2.dilate(edge_map, dilate_kernel, iterations=1)


def _candidate_from_contour(
    contour: np.ndarray,
    image_w: int,
    image_h: int,
    scale: float,
    method: str,
) -> ContourCandidate | None:
    contour_area = cv2.contourArea(contour)
    image_area = float(image_w * image_h)
    if image_area <= 0:
        return None

    area_ratio = contour_area / image_area
    if area_ratio < 0.08 or area_ratio > 0.88:
        return None

    quad = _quad_from_contour(contour)
    if quad is None:
        return None

    score = _score_quad(quad, image_w, image_h, contour_area)
    if score <= 0:
        return None

    return ContourCandidate(points=(quad / scale).astype("float32"), confidence=score, method=method)


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(contour, True)
    for epsilon_ratio in (0.015, 0.02, 0.03, 0.04, 0.06):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype("float32")
    if cv2.isContourConvex(box.reshape(-1, 1, 2)):
        return box
    return None


def _score_quad(quad: np.ndarray, image_w: int, image_h: int, contour_area: float) -> float:
    ordered = order_points(quad)
    (tl, tr, br, bl) = ordered
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    width = max(width_top, width_bottom)
    height = max(height_left, height_right)
    if width < image_w * 0.18 or height < image_h * 0.18:
        return 0.0

    aspect = max(width / max(height, 1.0), height / max(width, 1.0))
    if aspect > 3.2:
        return 0.0

    quad_area = cv2.contourArea(ordered.astype("float32"))
    area_ratio = quad_area / float(image_w * image_h)
    if area_ratio < 0.08 or area_ratio > 0.90:
        return 0.0

    _, _, w, h = cv2.boundingRect(ordered.astype("float32"))
    rectangularity = min(quad_area / max(float(w * h), 1.0), 1.0)
    side_balance_w = min(width_top, width_bottom) / max(width_top, width_bottom, 1.0)
    side_balance_h = min(height_left, height_right) / max(height_left, height_right, 1.0)
    balance = min(side_balance_w, side_balance_h)

    margin = min(
        ordered[:, 0].min(),
        ordered[:, 1].min(),
        image_w - ordered[:, 0].max(),
        image_h - ordered[:, 1].max(),
    )
    touches_frame_penalty = 0.12 if margin < 4 and area_ratio > 0.72 else 0.0

    raw_confidence = (
        min(area_ratio / 0.55, 1.0) * 0.35
        + rectangularity * 0.25
        + balance * 0.25
        + min(contour_area / max(quad_area, 1.0), 1.0) * 0.15
        - touches_frame_penalty
    )
    return float(np.clip(raw_confidence, 0.0, 0.99))
