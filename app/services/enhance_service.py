from app.services.scanner.enhancement import (
    SUPPORTED_ENHANCEMENT_MODES,
    apply_clahe,
    apply_levels,
    build_paper_mask,
    build_text_mask,
    correct_local_illumination,
    enhance_bw,
    enhance_color,
    enhance_document as enhance_document_result,
    enhance_gray,
    enhance_print,
    flatten_luminance,
    preserve_text,
    sharpen_image,
    whiten_paper_regions,
)


def enhance_document(image_bgr, mode: str = "auto"):
    """Compatibility wrapper returning only the enhanced image."""
    return enhance_document_result(image_bgr, mode).image
