import pytest
from httpx import AsyncClient, ASGITransport
import cv2
import numpy as np
import base64
import json
from io import BytesIO

from PIL import Image

from app.main import app
from app.api.v1.scan import (
    full_image_scan as full_image_endpoint,
    manual_crop_scan as manual_crop_endpoint,
    scan as scan_endpoint,
)
from app.core.exceptions import InvalidFileType
from app.services.pdf_service import (
    PDF_IMAGE_MAX_DIMENSION,
    _prepare_pdf_image,
    generate_pdf,
)
from app.config import settings


class UploadStub:
    def __init__(self, content: bytes, content_type: str):
        self._content = content
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._content


def make_upload_file(filename: str, content: bytes, content_type: str) -> UploadStub:
    return UploadStub(content, content_type)

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_scan_endpoint_success():
    # Programmatically create a simple image (e.g. 100x100 white rectangle on black background)
    img = np.zeros((200, 200, 3), dtype="uint8")
    cv2.rectangle(img, (20, 20), (180, 180), (255, 255, 255), -1)
    _, encoded = cv2.imencode(".png", img)
    img_bytes = encoded.tobytes()

    response = await scan_endpoint(make_upload_file("test.png", img_bytes, "image/png"))

    assert response.success is True
    assert response.image_base64
    assert response.image_mime_type == "image/jpeg"
    assert response.width > 0
    assert response.height > 0
    assert isinstance(response.edge_detected, bool)
    assert response.confidence >= 0
    assert response.crop_confidence >= 0
    assert response.crop_method
    assert response.background_whiteness >= 0
    assert response.shadow_score >= 0
    assert response.text_contrast >= 0
    assert response.blur_score >= 0
    assert response.glare_score >= 0
    assert response.selected_enhancement
    assert response.processing_mode == "auto"
    assert isinstance(response.warnings, list)

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "print", "color", "gray", "bw", "soft"])
async def test_scan_endpoint_accepts_enhancement_modes(mode):
    img = np.zeros((220, 180, 3), dtype="uint8")
    cv2.rectangle(img, (25, 25), (155, 195), (245, 245, 245), -1)
    _, encoded = cv2.imencode(".png", img)

    response = await scan_endpoint(make_upload_file("test.png", encoded.tobytes(), "image/png"), mode=mode)

    assert response.processing_mode == mode

@pytest.mark.asyncio
async def test_scan_endpoint_invalid_file_type():
    with pytest.raises(InvalidFileType):
        await scan_endpoint(make_upload_file("test.txt", b"plain text content", "text/plain"))


@pytest.mark.asyncio
async def test_manual_crop_endpoint_success():
    img = np.zeros((260, 220, 3), dtype="uint8")
    cv2.rectangle(img, (35, 25), (185, 235), (245, 245, 245), -1)
    cv2.putText(img, "DOC", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    _, encoded = cv2.imencode(".png", img)

    points = [
        {"x": 35 / 219, "y": 25 / 259},
        {"x": 185 / 219, "y": 25 / 259},
        {"x": 185 / 219, "y": 235 / 259},
        {"x": 35 / 219, "y": 235 / 259},
    ]
    response = await manual_crop_endpoint(
        make_upload_file("test.png", encoded.tobytes(), "image/png"),
        points_json=json.dumps(points),
    )

    assert response.success is True
    assert response.image_base64
    assert response.image_mime_type == "image/jpeg"
    assert response.width > 0
    assert response.height > 0
    assert response.edge_detected is True
    assert response.confidence == 1.0
    assert response.crop_confidence == 1.0
    assert response.crop_method == "manual_corners"
    assert response.processing_mode == "auto"


@pytest.mark.asyncio
async def test_full_image_endpoint_skips_auto_crop():
    img = np.zeros((260, 220, 3), dtype="uint8")
    cv2.rectangle(img, (35, 25), (185, 235), (245, 245, 245), -1)
    cv2.putText(img, "DOC", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    _, encoded = cv2.imencode(".png", img)

    response = await full_image_endpoint(make_upload_file("test.png", encoded.tobytes(), "image/png"))

    assert response.success is True
    assert response.image_base64
    assert response.image_mime_type == "image/jpeg"
    assert response.width == 220
    assert response.height == 260
    assert response.edge_detected is False
    assert response.confidence == 1.0
    assert response.crop_confidence == 1.0
    assert response.crop_method == "full_image_confirmed"


@pytest.mark.asyncio
async def test_manual_crop_accepts_camera_jpg_content_type():
    img = np.zeros((260, 220, 3), dtype="uint8")
    cv2.rectangle(img, (35, 25), (185, 235), (245, 245, 245), -1)
    _, encoded = cv2.imencode(".jpg", img)

    points = [
        {"x": 0.05, "y": 0.05},
        {"x": 0.95, "y": 0.05},
        {"x": 0.95, "y": 0.95},
        {"x": 0.05, "y": 0.95},
    ]
    response = await manual_crop_endpoint(
        make_upload_file("camera.jpg", encoded.tobytes(), "image/jpg"),
        points_json=json.dumps(points),
    )

    assert response.success is True
    assert response.crop_method == "manual_corners"


@pytest.mark.asyncio
async def test_full_image_endpoint_limits_large_camera_image_dimensions():
    img = np.full((3600, 2400, 3), 245, dtype="uint8")
    cv2.putText(img, "DOC", (300, 900), cv2.FONT_HERSHEY_SIMPLEX, 6, (25, 25, 25), 12)
    _, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    response = await full_image_endpoint(make_upload_file("large-camera.jpg", encoded.tobytes(), "image/jpeg"))

    assert response.success is True
    assert response.image_mime_type == "image/jpeg"
    assert max(response.width, response.height) == settings.scan_max_dimension


@pytest.mark.asyncio
async def test_pdf_generate_endpoint_success():
    # Make programmatically a dummy base64 string
    img = np.zeros((50, 50, 3), dtype="uint8")
    _, encoded = cv2.imencode(".png", img)
    dummy_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    
    payload = {
        "pages": [dummy_base64]
    }
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/pdf/generate", json=payload)
        
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 0


def test_pdf_generation_compresses_large_png_page():
    rng = np.random.default_rng(123)
    img = np.full((1600, 1200, 3), 238, dtype="uint8")
    texture = rng.normal(0, 8, img.shape[:2]).astype("int16")
    img = np.clip(img.astype("int16") + texture[:, :, None], 0, 255).astype("uint8")
    cv2.putText(img, "DOCUMENT SCAN", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
    for y in range(360, 1300, 100):
        cv2.line(img, (120, y), (1050, y), (45, 45, 45), 2)

    _, encoded = cv2.imencode(".png", img)
    page = base64.b64encode(encoded.tobytes()).decode("utf-8")

    pdf_bytes = generate_pdf([page])

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) < 900_000
    assert len(pdf_bytes) < len(encoded.tobytes())


def test_pdf_image_preparation_flattens_alpha_and_limits_dimensions():
    image = Image.new("RGBA", (2600, 1800), (255, 255, 255, 0))
    image.paste((40, 40, 40, 255), (200, 200, 2400, 500))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    page = base64.b64encode(buffer.getvalue()).decode("utf-8")

    jpeg_bytes = _prepare_pdf_image(f"data:image/png;base64,{page}")

    with Image.open(BytesIO(jpeg_bytes)) as prepared:
        assert prepared.format == "JPEG"
        assert prepared.mode == "RGB"
        assert max(prepared.size) == PDF_IMAGE_MAX_DIMENSION
