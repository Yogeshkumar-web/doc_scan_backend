import base64
import cv2
import json
import structlog
from fastapi import APIRouter, UploadFile, File, Form, Query
from typing import Annotated
from app.schemas.scan import ScanResponse
from app.core.exceptions import InvalidFileType, FileTooLarge, ProcessingFailed
from app.services.scanner import (
    SUPPORTED_ENHANCEMENT_MODES,
    process_document,
    process_document_with_corners,
    process_full_document,
)
from app.config import settings

router = APIRouter()
logger = structlog.get_logger()

def process_and_enhance_job(image_bytes: bytes, mode: str):
    """Runs the scanner pipeline and encodes the final page as base64 PNG."""
    result = process_document(image_bytes, mode)
    return encode_scan_result(result)


def process_manual_crop_job(image_bytes: bytes, points_json: str, mode: str):
    try:
        points = json.loads(points_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Crop points must be valid JSON") from exc

    if not isinstance(points, list):
        raise ValueError("Crop points must be a list")

    result = process_document_with_corners(image_bytes, points, mode)
    return encode_scan_result(result)


def process_full_image_job(image_bytes: bytes, mode: str):
    result = process_full_document(image_bytes, mode)
    return encode_scan_result(result)


def encode_scan_result(result):
    success, encoded_img = cv2.imencode(".png", result.image)
    if not success:
        raise ValueError("Failed to encode processed image to PNG")

    h, w = result.image.shape[:2]
    base64_str = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    return base64_str, w, h, result


async def read_validated_image(image: UploadFile) -> bytes:
    if image.content_type not in ["image/jpeg", "image/jpg", "image/png", "application/octet-stream", None, ""]:
        logger.warning("invalid_file_type", content_type=image.content_type)
        raise InvalidFileType()

    image_bytes = await image.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        logger.warning("file_too_large", size_mb=size_mb, max_limit=settings.max_upload_size_mb)
        raise FileTooLarge(f"File size exceeds limit of {settings.max_upload_size_mb}MB.")
    return image_bytes


def build_scan_response(base64_str: str, w: int, h: int, result, mode: str) -> ScanResponse:
    return ScanResponse(
        success=True,
        image_base64=base64_str,
        width=w,
        height=h,
        edge_detected=result.edge_detected,
        confidence=result.crop_confidence,
        crop_confidence=result.crop_confidence,
        crop_method=result.crop_method,
        background_whiteness=result.metrics.background_whiteness,
        shadow_score=result.metrics.shadow_score,
        text_contrast=result.metrics.text_contrast,
        blur_score=result.metrics.blur_score,
        glare_score=result.metrics.glare_score,
        selected_enhancement=result.metrics.selected_enhancement,
        processing_mode=mode,
        warnings=result.warnings,
    )


@router.post("/scan", response_model=ScanResponse)
async def scan(
    image: UploadFile = File(...),
    mode: Annotated[str, Query(pattern="^(auto|print|color|gray|bw|soft)$")] = "auto",
):
    if mode not in SUPPORTED_ENHANCEMENT_MODES:
        mode = "auto"

    image_bytes = await read_validated_image(image)

    try:
        base64_str, w, h, result = process_and_enhance_job(image_bytes, mode)
    except Exception as e:
        logger.error("processing_error", error=str(e), exc_info=True)
        raise ProcessingFailed(f"Failed to process image: {str(e)}")

    return build_scan_response(base64_str, w, h, result, mode)


@router.post("/scan/full-image", response_model=ScanResponse)
async def full_image_scan(
    image: UploadFile = File(...),
    mode: Annotated[str, Query(pattern="^(auto|print|color|gray|bw|soft)$")] = "auto",
):
    if mode not in SUPPORTED_ENHANCEMENT_MODES:
        mode = "auto"

    image_bytes = await read_validated_image(image)

    try:
        base64_str, w, h, result = process_full_image_job(image_bytes, mode)
    except Exception as e:
        logger.error("full_image_processing_error", error=str(e), exc_info=True)
        raise ProcessingFailed(f"Failed to process image: {str(e)}")

    return build_scan_response(base64_str, w, h, result, mode)


@router.post("/scan/manual-crop", response_model=ScanResponse)
async def manual_crop_scan(
    image: UploadFile = File(...),
    points_json: str = Form(...),
    mode: Annotated[str, Query(pattern="^(auto|print|color|gray|bw|soft)$")] = "auto",
):
    if mode not in SUPPORTED_ENHANCEMENT_MODES:
        mode = "auto"

    image_bytes = await read_validated_image(image)

    try:
        base64_str, w, h, result = process_manual_crop_job(image_bytes, points_json, mode)
    except Exception as e:
        logger.error("manual_crop_processing_error", error=str(e), exc_info=True)
        raise ProcessingFailed(f"Failed to process image: {str(e)}")

    return build_scan_response(base64_str, w, h, result, mode)
