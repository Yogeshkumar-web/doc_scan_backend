import cv2
import numpy as np

from app.services.scanner.models import QualityMetrics


def compute_quality_metrics(image_bgr: np.ndarray) -> QualityMetrics:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = compute_blur_score(gray)
    glare_score = compute_glare_score(gray)

    dark_ratio = float(np.mean(gray < 90))
    white_ratio = float(np.mean(gray > 235))
    if white_ratio > 0.70 and 0.002 < dark_ratio < 0.35:
        return QualityMetrics(
            background_whiteness=round(float(np.clip(white_ratio, 0.0, 1.0)), 2),
            shadow_score=0.0,
            text_contrast=1.0,
            blur_score=blur_score,
            glare_score=glare_score,
            binary_ink_ratio=round(dark_ratio, 3),
        )

    text_threshold = float(np.percentile(gray, 28))
    text_mask = gray <= text_threshold
    paper_mask = gray >= float(np.percentile(gray, 58))

    if paper_mask.any():
        paper_values = gray[paper_mask].astype("float32")
    else:
        paper_values = gray.astype("float32")

    if text_mask.any():
        text_values = gray[text_mask].astype("float32")
    else:
        text_values = gray.astype("float32")

    background_whiteness = float(np.clip(np.mean(paper_values) / 255.0, 0.0, 1.0))
    shadow_score = float(np.clip(np.std(paper_values) / 80.0, 0.0, 1.0))
    text_contrast = float(
        np.clip((np.mean(paper_values) - np.mean(text_values)) / 180.0, 0.0, 1.0)
    )
    return QualityMetrics(
        background_whiteness=round(background_whiteness, 2),
        shadow_score=round(shadow_score, 2),
        text_contrast=round(text_contrast, 2),
        blur_score=blur_score,
        glare_score=glare_score,
        binary_ink_ratio=round(dark_ratio, 3),
    )


def warnings_for_quality(metrics: QualityMetrics, mode: str) -> list[str]:
    warnings: list[str] = []
    if mode in {"auto", "print", "gray"} and metrics.background_whiteness < 0.78:
        warnings.append("LOW_BACKGROUND_WHITENESS")
    if metrics.shadow_score > 0.42:
        warnings.append("SHADOW_REMAINS")
    if metrics.text_contrast < 0.35:
        warnings.append("LOW_TEXT_CONTRAST")
    if metrics.blur_score > 0.72:
        warnings.append("IMAGE_BLURRY")
    if metrics.glare_score > 0.08:
        warnings.append("GLARE_DETECTED")
    return warnings


def compute_blur_score(gray: np.ndarray) -> float:
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Typical phone document images below ~55 are visibly soft.
    return round(float(np.clip(1.0 - laplacian_variance / 180.0, 0.0, 1.0)), 2)


def compute_glare_score(gray: np.ndarray) -> float:
    bright = gray > 245
    if not bright.any():
        return 0.0
    # Large bright areas are glare risk; small white paper areas are normal.
    bright_ratio = float(np.mean(bright))
    saturated_ratio = float(np.mean(gray > 252))
    return round(float(np.clip((saturated_ratio * 0.7) + max(0.0, bright_ratio - 0.35) * 0.3, 0.0, 1.0)), 2)


def with_selected_enhancement(metrics: QualityMetrics, selected: str) -> QualityMetrics:
    return QualityMetrics(
        background_whiteness=metrics.background_whiteness,
        shadow_score=metrics.shadow_score,
        text_contrast=metrics.text_contrast,
        blur_score=metrics.blur_score,
        glare_score=metrics.glare_score,
        binary_ink_ratio=metrics.binary_ink_ratio,
        selected_enhancement=selected,
    )
