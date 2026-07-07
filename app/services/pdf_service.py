import base64
from io import BytesIO

import img2pdf
from PIL import Image


PDF_IMAGE_MAX_DIMENSION = 2400
PDF_JPEG_QUALITY = 90


def _decode_base64_image(page: str) -> bytes:
    if "," in page:
        page = page.split(",", 1)[1]
    return base64.b64decode(page)


def _fit_for_pdf(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.convert("RGBA").getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        image = background
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    max_side = max(image.size)
    if max_side > PDF_IMAGE_MAX_DIMENSION:
        scale = PDF_IMAGE_MAX_DIMENSION / max_side
        new_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def _prepare_pdf_image(page: str) -> bytes:
    image_bytes = _decode_base64_image(page)
    with Image.open(BytesIO(image_bytes)) as image:
        image = _fit_for_pdf(image)
        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=PDF_JPEG_QUALITY,
            optimize=True,
        )
        return output.getvalue()


def generate_pdf(base64_pages: list[str]) -> bytes:
    image_bytes_list = [_prepare_pdf_image(page) for page in base64_pages]
    return img2pdf.convert(image_bytes_list)
