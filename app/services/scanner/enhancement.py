from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from app.services.scanner.models import EnhancementResult, QualityMetrics
from app.services.scanner.quality import compute_quality_metrics, warnings_for_quality, with_selected_enhancement

SUPPORTED_ENHANCEMENT_MODES = {"auto", "print", "color", "gray", "bw", "soft"}


@dataclass(frozen=True)
class AutoCandidate:
    name: str
    image: np.ndarray
    metrics: QualityMetrics
    warnings: list[str]
    score: float


def enhance_document(image_bgr: np.ndarray, mode: str = "auto") -> EnhancementResult:
    if mode not in SUPPORTED_ENHANCEMENT_MODES:
        mode = "auto"

    if mode == "auto":
        return enhance_auto(image_bgr)

    if mode == "print":
        enhanced = enhance_print(image_bgr)
    elif mode == "gray":
        enhanced = enhance_gray(image_bgr)
    elif mode == "bw":
        enhanced = enhance_bw(image_bgr)
    elif mode == "soft":
        enhanced = enhance_color(image_bgr, clip_limit=1.05, sharpness=0.0, flatten_strength=0.08)
    else:
        enhanced = enhance_color(image_bgr)

    metrics = with_selected_enhancement(compute_quality_metrics(enhanced), mode)
    return EnhancementResult(
        image=enhanced,
        metrics=metrics,
        warnings=warnings_for_quality(metrics, mode),
    )


def enhance_auto(image_bgr: np.ndarray) -> EnhancementResult:
    candidates = build_auto_candidates(image_bgr)
    chosen = max(candidates, key=lambda candidate: candidate.score)
    selected_warning = f"SELECTED_{chosen.name.upper()}"
    return EnhancementResult(
        image=chosen.image,
        metrics=with_selected_enhancement(chosen.metrics, chosen.name),
        warnings=sorted(set([*chosen.warnings, selected_warning]), key=[*chosen.warnings, selected_warning].index),
    )


def build_auto_candidates(image_bgr: np.ndarray) -> list[AutoCandidate]:
    clean_grayscale = enhance_clean_grayscale(image_bgr)
    clean_color = (
        clean_grayscale
        if is_low_saturation_document(image_bgr)
        else enhance_clean_color(image_bgr)
    )
    print_clean = enhance_print_clean(image_bgr, fallback_gray_bgr=clean_grayscale)
    candidate_images = [
        ("clean_color", clean_color),
        ("clean_grayscale", clean_grayscale),
        ("print_clean", print_clean),
    ]

    candidates: list[AutoCandidate] = []
    for name, image in candidate_images:
        metrics = with_selected_enhancement(compute_quality_metrics(image), name)
        warning_mode = "print" if name == "print_clean" else "gray"
        warnings = warnings_for_quality(metrics, warning_mode)
        score = score_auto_candidate(image, metrics, name)
        candidates.append(
            AutoCandidate(
                name=name,
                image=image,
                metrics=metrics,
                warnings=warnings,
                score=score,
            )
        )
    return candidates


def score_auto_candidate(image_bgr: np.ndarray, metrics: QualityMetrics, name: str) -> float:
    diagnostics = candidate_diagnostics(image_bgr)
    score = (
        metrics.background_whiteness * 0.27
        + metrics.text_contrast * 0.30
        + (1.0 - metrics.shadow_score) * 0.22
        + (1.0 - metrics.blur_score) * 0.06
        + (1.0 - metrics.glare_score) * 0.05
    )

    if name == "print_clean":
        score += 0.08
        if metrics.binary_ink_ratio < 0.004:
            score -= 0.55
        elif metrics.binary_ink_ratio < 0.012:
            score -= 0.18
        if metrics.binary_ink_ratio > 0.18:
            score -= 0.60
        elif metrics.binary_ink_ratio > 0.12:
            score -= 0.22
        if metrics.background_whiteness < 0.88:
            score -= 0.22
        if metrics.glare_score > 0.70:
            score -= 0.42
        elif metrics.glare_score > 0.40:
            score -= 0.22
        if diagnostics["mid_tone_ratio"] > 0.12:
            score -= 0.14
    elif name == "clean_grayscale":
        score += 0.10
        if diagnostics["white_ratio"] > 0.93 and metrics.text_contrast < 0.55:
            score -= 0.12
    elif name == "clean_color":
        score += 0.06
        if diagnostics["saturation_mean"] > 75 and metrics.background_whiteness < 0.84:
            score -= 0.08

    if diagnostics["dark_ratio"] > 0.22:
        score -= 0.20
    if diagnostics["ink_ratio"] > 0.18:
        score -= 0.18
    if diagnostics["median_stroke_width"] > 3.2:
        score -= min(0.22, (diagnostics["median_stroke_width"] - 3.2) * 0.06)
    if diagnostics["white_ratio"] > 0.96 and metrics.binary_ink_ratio < 0.008:
        score -= 0.25

    return score


def candidate_diagnostics(image_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    stroke_metrics = compute_stroke_metrics(gray)
    return {
        "dark_ratio": float(np.mean(gray < 70)),
        "ink_ratio": float(np.mean(gray < 105)),
        "white_ratio": float(np.mean(gray > 245)),
        "mid_tone_ratio": float(np.mean((gray > 70) & (gray < 225))),
        "saturation_mean": float(np.mean(hsv[:, :, 1])),
        "median_stroke_width": stroke_metrics["median_stroke_width"],
    }


def enhance_clean_color(image_bgr: np.ndarray) -> np.ndarray:
    return cleanup_document_color(image_bgr)


def is_low_saturation_document(image_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    paper_like = hsv[:, :, 2] > int(np.percentile(hsv[:, :, 2], 35))
    if not np.any(paper_like):
        return False

    return float(np.mean(saturation[paper_like])) < 40.0


def enhance_clean_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cleanup_document_gray(image_bgr), cv2.COLOR_GRAY2BGR)


def enhance_print_clean(image_bgr: np.ndarray, fallback_gray_bgr: np.ndarray | None = None) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    normalized = normalize_for_ink_extraction(gray)
    ink_mask = remove_border_artifacts(extract_ink_mask(normalized))
    if not binary_threshold_is_safe(gray, normalized, ink_mask):
        return fallback_gray_bgr if fallback_gray_bgr is not None else enhance_clean_grayscale(image_bgr)

    page = np.full_like(gray, 255)
    page[ink_mask > 0] = 0
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def cleanup_document_gray(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    text_mask = build_text_mask(gray)
    paper_mask = build_paper_mask(image_bgr, gray, text_mask)
    cleanup_mask = build_shadow_cleanup_mask(image_bgr, gray, text_mask, paper_mask)
    text_mask = refine_text_mask(text_mask)

    shadow_removed = correct_local_illumination(gray, strength=0.82, lift=8)
    paper_cleaned = blend_with_mask(gray, shadow_removed, cleanup_mask)
    paper_cleaned = remove_cast_shadows(paper_cleaned, cleanup_mask, text_mask, strength=0.86)
    paper_cleaned = normalize_paper_luminance(paper_cleaned, cleanup_mask, strength=0.72)
    paper_cleaned = whiten_paper_regions(paper_cleaned, cleanup_mask, threshold=105.0, strength=0.92)
    paper_cleaned = flatten_paper_variation(paper_cleaned, cleanup_mask, strength=0.46)
    paper_cleaned = apply_clahe(paper_cleaned, clip_limit=1.02)
    paper_cleaned = sharpen_image(paper_cleaned, amount=0.04, radius=0.8)
    return preserve_text(paper_cleaned, gray, text_mask)


def cleanup_document_color(image_bgr: np.ndarray) -> np.ndarray:
    denoised = cv2.bilateralFilter(image_bgr, 5, 25, 25)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    text_mask = build_text_mask(gray)
    paper_mask = build_paper_mask(denoised, gray, text_mask)
    cleanup_mask = build_shadow_cleanup_mask(denoised, gray, text_mask, paper_mask)
    text_mask = refine_text_mask(text_mask)

    shadow_removed = correct_local_illumination(l, strength=0.68, lift=6)
    l_cleaned = blend_with_mask(l, shadow_removed, cleanup_mask)
    l_cleaned = remove_cast_shadows(l_cleaned, cleanup_mask, text_mask, strength=0.72)
    l_cleaned = normalize_paper_luminance(l_cleaned, cleanup_mask, strength=0.54)
    l_cleaned = whiten_paper_regions(l_cleaned, cleanup_mask, threshold=112.0, strength=0.78)
    l_cleaned = flatten_paper_variation(l_cleaned, cleanup_mask, strength=0.32)
    l_cleaned = apply_clahe(l_cleaned, clip_limit=1.03)
    l_cleaned = sharpen_image(l_cleaned, amount=0.03, radius=0.9)
    l_cleaned = preserve_text(l_cleaned, l, text_mask)

    a_cleaned = neutralize_paper_chroma(a, cleanup_mask, strength=0.40)
    b_cleaned = neutralize_paper_chroma(b, cleanup_mask, strength=0.40)
    return cv2.cvtColor(cv2.merge((l_cleaned, a_cleaned, b_cleaned)), cv2.COLOR_LAB2BGR)


def enhance_print(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    normalized = normalize_for_ink_extraction(gray)
    ink_mask = extract_ink_mask(normalized)
    ink_mask = remove_border_artifacts(ink_mask)
    page = np.full_like(gray, 255)
    page[ink_mask > 0] = 0
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def binary_threshold_is_safe(gray: np.ndarray, normalized_gray: np.ndarray, ink_mask: np.ndarray) -> bool:
    ink_ratio = float(np.mean(ink_mask > 0))
    if ink_ratio < 0.006 or ink_ratio > 0.12:
        return False

    text_mask = build_text_mask(gray)
    paper_mask = build_paper_mask(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, text_mask)
    paper_ratio = float(np.mean(paper_mask > 80))
    if paper_ratio < 0.42:
        return False

    paper_values = normalized_gray[paper_mask > 80]
    original_paper_values = gray[paper_mask > 80]
    if paper_values.size == 0 or original_paper_values.size == 0:
        return False

    paper_mean = float(np.mean(paper_values))
    paper_std = float(np.std(paper_values))
    original_paper_std = float(np.std(original_paper_values))
    original_paper_dark_ratio = float(np.mean(original_paper_values < 150))

    # ROOT CAUSE FIX: a ruled table's border lines form one large connected
    # component spanning most of the page, so largest_component_ratio alone
    # was flagging every table/form document as "unsafe" and silently
    # discarding the crisp print_clean candidate for grayscale fallback.
    #
    # Distinguish real failure blobs (illumination/threshold breakdown -->
    # a dense, filled dark region) from grid/table lines (large footprint,
    # but low fill density inside their own bounding box) using density.
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    largest_component_ratio = 0.0
    largest_component_density = 0.0
    if component_count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1
        largest_area = float(stats[largest_idx, cv2.CC_STAT_AREA])
        bbox_area = float(
            stats[largest_idx, cv2.CC_STAT_WIDTH] * stats[largest_idx, cv2.CC_STAT_HEIGHT]
        )
        total_area = ink_mask.shape[0] * ink_mask.shape[1]
        largest_component_ratio = largest_area / max(total_area, 1)
        largest_component_density = largest_area / max(bbox_area, 1)

    # Large + dense => genuine failure (e.g. a whole dark shadow region).
    # Large + sparse => grid/table lines, which is fine to keep.
    is_dense_failure_blob = largest_component_ratio > 0.035 and largest_component_density >= 0.15

    return (
        paper_mean >= 175.0
        and paper_std <= 42.0
        and original_paper_std <= 50.0
        and original_paper_dark_ratio <= 0.18
        and not is_dense_failure_blob
    )


def enhance_gray(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = correct_local_illumination(gray, strength=0.75, lift=8)
    gray = flatten_luminance(gray, strength=0.13, lift=10)
    gray = apply_clahe(gray, clip_limit=1.1)
    gray = apply_levels(gray, low_percentile=2.0, high_percentile=88.0)
    gray = sharpen_image(gray, amount=0.12, radius=0.8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def enhance_color(
    image_bgr: np.ndarray,
    clip_limit: float = 1.15,
    sharpness: float = 0.12,
    flatten_strength: float = 0.10,
) -> np.ndarray:
    denoised = cv2.bilateralFilter(image_bgr, 5, 25, 25)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = correct_local_illumination(l, strength=0.55, lift=6)
    l = flatten_luminance(l, strength=flatten_strength, lift=8)
    l = apply_clahe(l, clip_limit=clip_limit)
    if sharpness > 0:
        l = sharpen_image(l, amount=sharpness, radius=0.9)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance_bw(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = correct_local_illumination(gray, strength=0.95, lift=12)
    gray = apply_clahe(gray, clip_limit=1.1)

    block_size = max(35, int(min(gray.shape[:2]) / 18) | 1)
    block_size = min(block_size, 91)
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        8,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def normalize_for_ink_extraction(gray: np.ndarray) -> np.ndarray:
    """Normalize uneven light for binarization without keeping paper texture."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype("uint8")


def extract_ink_mask(normalized_gray: np.ndarray) -> np.ndarray:
    window_size = max(35, int(min(normalized_gray.shape[:2]) / 16) | 1)
    window_size = min(window_size, 75)
    if window_size % 2 == 0:
        window_size += 1

    threshold = threshold_sauvola(normalized_gray, window_size=window_size, k=0.18, r=128)
    binary_page = (normalized_gray > threshold).astype("uint8") * 255
    ink_mask = cv2.bitwise_not(binary_page)

    # Remove pinhole noise but keep text/handwriting strokes.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return remove_large_mask_components(ink_mask, max_area_ratio=0.035)


def remove_border_artifacts(ink_mask: np.ndarray) -> np.ndarray:
    cleaned = ink_mask.copy()
    h, w = cleaned.shape[:2]
    strip_w = max(4, int(w * 0.018))
    strip_h = max(4, int(h * 0.012))

    strips = [
        (slice(None), slice(0, strip_w)),
        (slice(None), slice(w - strip_w, w)),
        (slice(0, strip_h), slice(None)),
        (slice(h - strip_h, h), slice(None)),
    ]
    for ys, xs in strips:
        strip = cleaned[ys, xs]
        black_ratio = float(np.count_nonzero(strip)) / max(strip.size, 1)
        if black_ratio > 0.22:
            cleaned[ys, xs] = 0
    return cleaned


def odd_kernel(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def background_kernel_size(image: np.ndarray) -> int:
    shorter_side = min(image.shape[:2])
    return odd_kernel(max(51, min(181, int(shorter_side * 0.12))))


def correct_local_illumination(gray: np.ndarray, strength: float = 0.85, lift: int = 8) -> np.ndarray:
    kernel_size = background_kernel_size(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(9, kernel_size / 6))
    target = float(np.percentile(background, 88))
    correction = (target - background.astype("float32")) * strength
    corrected = gray.astype("float32") + correction + lift
    return np.clip(corrected, 0, 255).astype("uint8")


def flatten_luminance(gray: np.ndarray, strength: float = 0.16, lift: int = 10) -> np.ndarray:
    background = cv2.medianBlur(gray, background_kernel_size(gray))
    flattened = cv2.addWeighted(gray, 1.0 + strength, background, -strength, lift)
    return np.clip(flattened, 0, 255).astype("uint8")


def apply_levels(
    gray: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 84.0,
) -> np.ndarray:
    low = float(np.percentile(gray, low_percentile))
    high = float(np.percentile(gray, high_percentile))
    leveled = (gray.astype("float32") - low) * 255.0 / max(high - low, 1.0)
    return np.clip(leveled, 0, 255).astype("uint8")


def apply_clahe(gray: np.ndarray, clip_limit: float = 1.1) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sharpen_image(image: np.ndarray, amount: float = 0.12, radius: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def build_text_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2)
    local_dark = cv2.subtract(blurred, gray)
    local_strokes = cv2.inRange(local_dark, 8, 255)
    dark_pixels = cv2.inRange(gray, 0, int(np.percentile(gray, 24)))
    text_mask = cv2.bitwise_and(local_strokes, dark_pixels)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return remove_large_mask_components(text_mask, max_area_ratio=0.018)


def expand_text_mask(text_mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(text_mask, kernel, iterations=1)


def refine_text_mask(text_mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def compute_stroke_metrics(gray: np.ndarray) -> dict[str, float]:
    threshold = min(125, int(np.percentile(gray, 22)))
    ink = cv2.inRange(gray, 0, threshold)
    ink = remove_large_mask_components(ink, max_area_ratio=0.025)
    if not np.any(ink):
        return {"median_stroke_width": 0.0}

    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    widths = distance[ink > 0] * 2.0
    if widths.size == 0:
        return {"median_stroke_width": 0.0}
    return {"median_stroke_width": float(np.median(widths))}


def remove_large_mask_components(mask: np.ndarray, max_area_ratio: float) -> np.ndarray:
    total_area = mask.shape[0] * mask.shape[1]
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return np.zeros_like(mask)

    max_area = total_area * max_area_ratio
    keep_labels = stats[:, cv2.CC_STAT_AREA] <= max_area
    keep_labels[0] = False
    return (keep_labels[labels].astype("uint8") * 255)


def build_paper_mask(image_bgr: np.ndarray, gray: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    low_sat = cv2.inRange(saturation, 0, 125)
    bright = cv2.inRange(value, max(95, int(np.percentile(value, 35))), 255)
    gray_bright = cv2.inRange(gray, int(np.percentile(gray, 35)), 255)
    paper_mask = cv2.bitwise_and(cv2.bitwise_and(low_sat, bright), gray_bright)
    paper_mask = cv2.bitwise_and(paper_mask, cv2.bitwise_not(text_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    paper_mask = cv2.GaussianBlur(paper_mask, (0, 0), sigmaX=5)
    return paper_mask


def build_shadow_cleanup_mask(
    image_bgr: np.ndarray,
    gray: np.ndarray,
    text_mask: np.ndarray,
    paper_mask: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    low_saturation = cv2.inRange(saturation, 0, 135)
    not_too_dark = cv2.inRange(value, 62, 255)
    shadow_candidate = cv2.bitwise_and(low_saturation, not_too_dark)
    shadow_candidate = cv2.bitwise_and(shadow_candidate, cv2.bitwise_not(text_mask))

    combined = cv2.bitwise_or(paper_mask, shadow_candidate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=8)
    return combined


def blend_with_mask(base: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weight = (mask.astype("float32") / 255.0) ** 0.85
    out = base.astype("float32") * (1.0 - weight) + cleaned.astype("float32") * weight
    return np.clip(out, 0, 255).astype("uint8")


def flatten_paper_variation(gray: np.ndarray, paper_mask: np.ndarray, strength: float = 0.25) -> np.ndarray:
    paper_pixels = gray[paper_mask > 80]
    if paper_pixels.size == 0:
        return gray

    target = float(np.percentile(paper_pixels, 88))
    x = gray.astype("float32")
    paper_weight = (paper_mask.astype("float32") / 255.0) ** 1.25
    flattened = x + (target - x) * paper_weight * strength
    return np.clip(flattened, 0, 255).astype("uint8")


def remove_cast_shadows(
    gray: np.ndarray,
    cleanup_mask: np.ndarray,
    text_mask: np.ndarray,
    strength: float = 0.82,
) -> np.ndarray:
    paper_pixels = gray[cleanup_mask > 80]
    if paper_pixels.size == 0:
        return gray

    kernel_size = odd_kernel(max(91, min(301, int(min(gray.shape[:2]) * 0.20))))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(18, kernel_size / 4))
    target = float(np.percentile(paper_pixels, 92))
    deficit = np.clip(target - background.astype("float32"), 0.0, 110.0)
    shadow_weight = np.clip(deficit / 85.0, 0.0, 1.0)
    cleanup_weight = (cleanup_mask.astype("float32") / 255.0) ** 0.90
    text_protection = 1.0 - (cv2.GaussianBlur(text_mask, (0, 0), sigmaX=1.0).astype("float32") / 255.0)
    correction = deficit * shadow_weight * cleanup_weight * text_protection * strength
    lifted = gray.astype("float32") + correction
    return np.clip(lifted, 0, 255).astype("uint8")


def normalize_paper_luminance(gray: np.ndarray, paper_mask: np.ndarray, strength: float = 0.55) -> np.ndarray:
    paper_pixels = gray[paper_mask > 80]
    if paper_pixels.size == 0:
        return gray

    kernel_size = background_kernel_size(gray)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(12, kernel_size / 5))
    target = float(np.percentile(paper_pixels, 88))
    normalized = gray.astype("float32") * target / np.maximum(background.astype("float32"), 1.0)
    weight = (paper_mask.astype("float32") / 255.0) ** 1.10
    out = gray.astype("float32") * (1.0 - weight * strength) + normalized * weight * strength
    return np.clip(out, 0, 255).astype("uint8")


def neutralize_paper_chroma(channel: np.ndarray, paper_mask: np.ndarray, strength: float = 0.35) -> np.ndarray:
    x = channel.astype("float32")
    paper_weight = (paper_mask.astype("float32") / 255.0) ** 1.15
    neutralized = x * (1.0 - paper_weight * strength) + 128.0 * paper_weight * strength
    return np.clip(neutralized, 0, 255).astype("uint8")


def whiten_paper_regions(
    gray: np.ndarray,
    paper_mask: np.ndarray,
    threshold: float = 150.0,
    strength: float = 0.90,
) -> np.ndarray:
    x = gray.astype("float32")
    tonal_mask = np.clip((x - threshold) / (255.0 - threshold), 0.0, 1.0)
    paper_weight = (paper_mask.astype("float32") / 255.0) * tonal_mask
    target = 255.0 - (255.0 - x) * (1.0 - strength)
    boosted = x * (1.0 - paper_weight) + target * paper_weight
    return np.clip(boosted, 0, 255).astype("uint8")


def preserve_text(enhanced: np.ndarray, source_gray: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    text_weight = cv2.GaussianBlur(text_mask, (0, 0), sigmaX=0.65).astype("float32") / 255.0
    source = source_gray.astype("float32")
    enhanced_f = enhanced.astype("float32")
    soft_darkening = np.clip((150.0 - source) / 150.0, 0.0, 1.0)
    text_source = source - (soft_darkening * 28.0)
    text_source = np.minimum(text_source, enhanced_f + 6.0)
    text_source = np.clip(text_source, 0, 210)
    out = enhanced_f * (1.0 - text_weight) + text_source * text_weight
    return np.clip(out, 0, 255).astype("uint8")

# from dataclasses import dataclass

# import cv2
# import numpy as np
# from skimage.filters import threshold_sauvola

# from app.services.scanner.models import EnhancementResult, QualityMetrics
# from app.services.scanner.quality import compute_quality_metrics, warnings_for_quality, with_selected_enhancement

# SUPPORTED_ENHANCEMENT_MODES = {"auto", "print", "color", "gray", "bw", "soft"}


# @dataclass(frozen=True)
# class AutoCandidate:
#     name: str
#     image: np.ndarray
#     metrics: QualityMetrics
#     warnings: list[str]
#     score: float


# def enhance_document(image_bgr: np.ndarray, mode: str = "auto") -> EnhancementResult:
#     if mode not in SUPPORTED_ENHANCEMENT_MODES:
#         mode = "auto"

#     if mode == "auto":
#         return enhance_auto(image_bgr)

#     if mode == "print":
#         enhanced = enhance_print(image_bgr)
#     elif mode == "gray":
#         enhanced = enhance_gray(image_bgr)
#     elif mode == "bw":
#         enhanced = enhance_bw(image_bgr)
#     elif mode == "soft":
#         enhanced = enhance_color(image_bgr, clip_limit=1.05, sharpness=0.0, flatten_strength=0.08)
#     else:
#         enhanced = enhance_color(image_bgr)

#     metrics = with_selected_enhancement(compute_quality_metrics(enhanced), mode)
#     return EnhancementResult(
#         image=enhanced,
#         metrics=metrics,
#         warnings=warnings_for_quality(metrics, mode),
#     )


# def enhance_auto(image_bgr: np.ndarray) -> EnhancementResult:
#     candidates = build_auto_candidates(image_bgr)
#     chosen = max(candidates, key=lambda candidate: candidate.score)
#     selected_warning = f"SELECTED_{chosen.name.upper()}"
#     return EnhancementResult(
#         image=chosen.image,
#         metrics=with_selected_enhancement(chosen.metrics, chosen.name),
#         warnings=sorted(set([*chosen.warnings, selected_warning]), key=[*chosen.warnings, selected_warning].index),
#     )


# def build_auto_candidates(image_bgr: np.ndarray) -> list[AutoCandidate]:
#     candidate_images = [
#         ("clean_color", enhance_clean_color(image_bgr)),
#         ("clean_grayscale", enhance_clean_grayscale(image_bgr)),
#         ("print_clean", enhance_print_clean(image_bgr)),
#     ]

#     candidates: list[AutoCandidate] = []
#     for name, image in candidate_images:
#         metrics = with_selected_enhancement(compute_quality_metrics(image), name)
#         warning_mode = "print" if name == "print_clean" else "gray"
#         warnings = warnings_for_quality(metrics, warning_mode)
#         score = score_auto_candidate(image, metrics, name)
#         candidates.append(
#             AutoCandidate(
#                 name=name,
#                 image=image,
#                 metrics=metrics,
#                 warnings=warnings,
#                 score=score,
#             )
#         )
#     return candidates


# def score_auto_candidate(image_bgr: np.ndarray, metrics: QualityMetrics, name: str) -> float:
#     diagnostics = candidate_diagnostics(image_bgr)
#     score = (
#         metrics.background_whiteness * 0.27
#         + metrics.text_contrast * 0.30
#         + (1.0 - metrics.shadow_score) * 0.22
#         + (1.0 - metrics.blur_score) * 0.06
#         + (1.0 - metrics.glare_score) * 0.05
#     )

#     if name == "print_clean":
#         score += 0.08
#         if metrics.binary_ink_ratio < 0.004:
#             score -= 0.55
#         elif metrics.binary_ink_ratio < 0.012:
#             score -= 0.18
#         if metrics.binary_ink_ratio > 0.18:
#             score -= 0.60
#         elif metrics.binary_ink_ratio > 0.12:
#             score -= 0.22
#         if metrics.background_whiteness < 0.88:
#             score -= 0.22
#         if metrics.glare_score > 0.70:
#             score -= 0.42
#         elif metrics.glare_score > 0.40:
#             score -= 0.22
#         if diagnostics["mid_tone_ratio"] > 0.12:
#             score -= 0.14
#     elif name == "clean_grayscale":
#         score += 0.10
#         if diagnostics["white_ratio"] > 0.93 and metrics.text_contrast < 0.55:
#             score -= 0.12
#     elif name == "clean_color":
#         score += 0.06
#         if diagnostics["saturation_mean"] > 75 and metrics.background_whiteness < 0.84:
#             score -= 0.08

#     if diagnostics["dark_ratio"] > 0.22:
#         score -= 0.20
#     if diagnostics["ink_ratio"] > 0.18:
#         score -= 0.18
#     if diagnostics["median_stroke_width"] > 3.2:
#         score -= min(0.22, (diagnostics["median_stroke_width"] - 3.2) * 0.06)
#     if diagnostics["white_ratio"] > 0.96 and metrics.binary_ink_ratio < 0.008:
#         score -= 0.25

#     return score


# def candidate_diagnostics(image_bgr: np.ndarray) -> dict[str, float]:
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
#     stroke_metrics = compute_stroke_metrics(gray)
#     return {
#         "dark_ratio": float(np.mean(gray < 70)),
#         "ink_ratio": float(np.mean(gray < 105)),
#         "white_ratio": float(np.mean(gray > 245)),
#         "mid_tone_ratio": float(np.mean((gray > 70) & (gray < 225))),
#         "saturation_mean": float(np.mean(hsv[:, :, 1])),
#         "median_stroke_width": stroke_metrics["median_stroke_width"],
#     }


# def enhance_clean_color(image_bgr: np.ndarray) -> np.ndarray:
#     return cleanup_document_color(image_bgr)


# def enhance_clean_grayscale(image_bgr: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(cleanup_document_gray(image_bgr), cv2.COLOR_GRAY2BGR)


# def enhance_print_clean(image_bgr: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 3)
#     normalized = normalize_for_ink_extraction(gray)
#     ink_mask = remove_border_artifacts(extract_ink_mask(normalized))
#     if not binary_threshold_is_safe(gray, normalized, ink_mask):
#         return enhance_clean_grayscale(image_bgr)

#     page = np.full_like(gray, 255)
#     page[ink_mask > 0] = 0
#     return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


# def cleanup_document_gray(image_bgr: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 3)
#     text_mask = build_text_mask(gray)
#     paper_mask = build_paper_mask(image_bgr, gray, text_mask)
#     cleanup_mask = build_shadow_cleanup_mask(image_bgr, gray, text_mask, paper_mask)
#     text_mask = refine_text_mask(text_mask)

#     shadow_removed = correct_local_illumination(gray, strength=0.82, lift=8)
#     paper_cleaned = blend_with_mask(gray, shadow_removed, cleanup_mask)
#     paper_cleaned = remove_cast_shadows(paper_cleaned, cleanup_mask, text_mask, strength=0.86)
#     paper_cleaned = normalize_paper_luminance(paper_cleaned, cleanup_mask, strength=0.72)
#     paper_cleaned = whiten_paper_regions(paper_cleaned, cleanup_mask, threshold=105.0, strength=0.92)
#     paper_cleaned = flatten_paper_variation(paper_cleaned, cleanup_mask, strength=0.46)
#     paper_cleaned = apply_clahe(paper_cleaned, clip_limit=1.02)
#     paper_cleaned = sharpen_image(paper_cleaned, amount=0.04, radius=0.8)
#     return preserve_text(paper_cleaned, gray, text_mask)


# def cleanup_document_color(image_bgr: np.ndarray) -> np.ndarray:
#     denoised = cv2.bilateralFilter(image_bgr, 5, 25, 25)
#     lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
#     text_mask = build_text_mask(gray)
#     paper_mask = build_paper_mask(denoised, gray, text_mask)
#     cleanup_mask = build_shadow_cleanup_mask(denoised, gray, text_mask, paper_mask)
#     text_mask = refine_text_mask(text_mask)

#     shadow_removed = correct_local_illumination(l, strength=0.68, lift=6)
#     l_cleaned = blend_with_mask(l, shadow_removed, cleanup_mask)
#     l_cleaned = remove_cast_shadows(l_cleaned, cleanup_mask, text_mask, strength=0.72)
#     l_cleaned = normalize_paper_luminance(l_cleaned, cleanup_mask, strength=0.54)
#     l_cleaned = whiten_paper_regions(l_cleaned, cleanup_mask, threshold=112.0, strength=0.78)
#     l_cleaned = flatten_paper_variation(l_cleaned, cleanup_mask, strength=0.32)
#     l_cleaned = apply_clahe(l_cleaned, clip_limit=1.03)
#     l_cleaned = sharpen_image(l_cleaned, amount=0.03, radius=0.9)
#     l_cleaned = preserve_text(l_cleaned, l, text_mask)

#     a_cleaned = neutralize_paper_chroma(a, cleanup_mask, strength=0.40)
#     b_cleaned = neutralize_paper_chroma(b, cleanup_mask, strength=0.40)
#     return cv2.cvtColor(cv2.merge((l_cleaned, a_cleaned, b_cleaned)), cv2.COLOR_LAB2BGR)


# def enhance_print(image_bgr: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 3)
#     normalized = normalize_for_ink_extraction(gray)
#     ink_mask = extract_ink_mask(normalized)
#     ink_mask = remove_border_artifacts(ink_mask)
#     page = np.full_like(gray, 255)
#     page[ink_mask > 0] = 0
#     return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


# def binary_threshold_is_safe(gray: np.ndarray, normalized_gray: np.ndarray, ink_mask: np.ndarray) -> bool:
#     ink_ratio = float(np.mean(ink_mask > 0))
#     if ink_ratio < 0.006 or ink_ratio > 0.12:
#         return False

#     text_mask = build_text_mask(gray)
#     paper_mask = build_paper_mask(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, text_mask)
#     paper_ratio = float(np.mean(paper_mask > 80))
#     if paper_ratio < 0.42:
#         return False

#     paper_values = normalized_gray[paper_mask > 80]
#     original_paper_values = gray[paper_mask > 80]
#     if paper_values.size == 0 or original_paper_values.size == 0:
#         return False

#     paper_mean = float(np.mean(paper_values))
#     paper_std = float(np.std(paper_values))
#     original_paper_std = float(np.std(original_paper_values))
#     original_paper_dark_ratio = float(np.mean(original_paper_values < 150))
#     component_count, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
#     total_area = ink_mask.shape[0] * ink_mask.shape[1]
#     largest_component_ratio = 0.0
#     if component_count > 1:
#         largest_component_ratio = float(np.max(stats[1:, cv2.CC_STAT_AREA])) / max(total_area, 1)

#     return (
#         paper_mean >= 175.0
#         and paper_std <= 42.0
#         and original_paper_std <= 50.0
#         and original_paper_dark_ratio <= 0.18
#         and largest_component_ratio <= 0.035
#     )


# def enhance_gray(image_bgr: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 3)
#     gray = correct_local_illumination(gray, strength=0.75, lift=8)
#     gray = flatten_luminance(gray, strength=0.13, lift=10)
#     gray = apply_clahe(gray, clip_limit=1.1)
#     gray = apply_levels(gray, low_percentile=2.0, high_percentile=88.0)
#     gray = sharpen_image(gray, amount=0.12, radius=0.8)
#     return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# def enhance_color(
#     image_bgr: np.ndarray,
#     clip_limit: float = 1.15,
#     sharpness: float = 0.12,
#     flatten_strength: float = 0.10,
# ) -> np.ndarray:
#     denoised = cv2.bilateralFilter(image_bgr, 5, 25, 25)
#     lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     l = correct_local_illumination(l, strength=0.55, lift=6)
#     l = flatten_luminance(l, strength=flatten_strength, lift=8)
#     l = apply_clahe(l, clip_limit=clip_limit)
#     if sharpness > 0:
#         l = sharpen_image(l, amount=sharpness, radius=0.9)
#     return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# def enhance_bw(image_bgr: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 3)
#     gray = correct_local_illumination(gray, strength=0.95, lift=12)
#     gray = apply_clahe(gray, clip_limit=1.1)

#     block_size = max(35, int(min(gray.shape[:2]) / 18) | 1)
#     block_size = min(block_size, 91)
#     if block_size % 2 == 0:
#         block_size += 1

#     binary = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         block_size,
#         8,
#     )
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
#     return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# def normalize_for_ink_extraction(gray: np.ndarray) -> np.ndarray:
#     """Normalize uneven light for binarization without keeping paper texture."""
#     background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
#     normalized = cv2.divide(gray, background, scale=255)
#     normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
#     return normalized.astype("uint8")


# def extract_ink_mask(normalized_gray: np.ndarray) -> np.ndarray:
#     window_size = max(35, int(min(normalized_gray.shape[:2]) / 16) | 1)
#     window_size = min(window_size, 75)
#     if window_size % 2 == 0:
#         window_size += 1

#     threshold = threshold_sauvola(normalized_gray, window_size=window_size, k=0.18, r=128)
#     binary_page = (normalized_gray > threshold).astype("uint8") * 255
#     ink_mask = cv2.bitwise_not(binary_page)

#     # Remove pinhole noise but keep text/handwriting strokes.
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     return remove_large_mask_components(ink_mask, max_area_ratio=0.035)


# def remove_border_artifacts(ink_mask: np.ndarray) -> np.ndarray:
#     cleaned = ink_mask.copy()
#     h, w = cleaned.shape[:2]
#     strip_w = max(4, int(w * 0.018))
#     strip_h = max(4, int(h * 0.012))

#     strips = [
#         (slice(None), slice(0, strip_w)),
#         (slice(None), slice(w - strip_w, w)),
#         (slice(0, strip_h), slice(None)),
#         (slice(h - strip_h, h), slice(None)),
#     ]
#     for ys, xs in strips:
#         strip = cleaned[ys, xs]
#         black_ratio = float(np.count_nonzero(strip)) / max(strip.size, 1)
#         if black_ratio > 0.22:
#             cleaned[ys, xs] = 0
#     return cleaned


# def odd_kernel(value: int) -> int:
#     return value if value % 2 == 1 else value + 1


# def background_kernel_size(image: np.ndarray) -> int:
#     shorter_side = min(image.shape[:2])
#     return odd_kernel(max(51, min(181, int(shorter_side * 0.12))))


# def correct_local_illumination(gray: np.ndarray, strength: float = 0.85, lift: int = 8) -> np.ndarray:
#     kernel_size = background_kernel_size(gray)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
#     background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(9, kernel_size / 6))
#     target = float(np.percentile(background, 88))
#     correction = (target - background.astype("float32")) * strength
#     corrected = gray.astype("float32") + correction + lift
#     return np.clip(corrected, 0, 255).astype("uint8")


# def flatten_luminance(gray: np.ndarray, strength: float = 0.16, lift: int = 10) -> np.ndarray:
#     background = cv2.medianBlur(gray, background_kernel_size(gray))
#     flattened = cv2.addWeighted(gray, 1.0 + strength, background, -strength, lift)
#     return np.clip(flattened, 0, 255).astype("uint8")


# def apply_levels(
#     gray: np.ndarray,
#     low_percentile: float = 2.0,
#     high_percentile: float = 84.0,
# ) -> np.ndarray:
#     low = float(np.percentile(gray, low_percentile))
#     high = float(np.percentile(gray, high_percentile))
#     leveled = (gray.astype("float32") - low) * 255.0 / max(high - low, 1.0)
#     return np.clip(leveled, 0, 255).astype("uint8")


# def apply_clahe(gray: np.ndarray, clip_limit: float = 1.1) -> np.ndarray:
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
#     return clahe.apply(gray)


# def sharpen_image(image: np.ndarray, amount: float = 0.12, radius: float = 0.8) -> np.ndarray:
#     blurred = cv2.GaussianBlur(image, (0, 0), radius)
#     return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


# def build_text_mask(gray: np.ndarray) -> np.ndarray:
#     blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2)
#     local_dark = cv2.subtract(blurred, gray)
#     local_strokes = cv2.inRange(local_dark, 8, 255)
#     dark_pixels = cv2.inRange(gray, 0, int(np.percentile(gray, 24)))
#     text_mask = cv2.bitwise_and(local_strokes, dark_pixels)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     return remove_large_mask_components(text_mask, max_area_ratio=0.018)


# def expand_text_mask(text_mask: np.ndarray) -> np.ndarray:
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     return cv2.dilate(text_mask, kernel, iterations=1)


# def refine_text_mask(text_mask: np.ndarray) -> np.ndarray:
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     return cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


# def compute_stroke_metrics(gray: np.ndarray) -> dict[str, float]:
#     threshold = min(125, int(np.percentile(gray, 22)))
#     ink = cv2.inRange(gray, 0, threshold)
#     ink = remove_large_mask_components(ink, max_area_ratio=0.025)
#     if not np.any(ink):
#         return {"median_stroke_width": 0.0}

#     distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
#     widths = distance[ink > 0] * 2.0
#     if widths.size == 0:
#         return {"median_stroke_width": 0.0}
#     return {"median_stroke_width": float(np.median(widths))}


# def remove_large_mask_components(mask: np.ndarray, max_area_ratio: float) -> np.ndarray:
#     total_area = mask.shape[0] * mask.shape[1]
#     component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
#     cleaned = np.zeros_like(mask)
#     for label in range(1, component_count):
#         area = stats[label, cv2.CC_STAT_AREA]
#         if area <= total_area * max_area_ratio:
#             cleaned[labels == label] = 255
#     return cleaned


# def build_paper_mask(image_bgr: np.ndarray, gray: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
#     hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
#     _, saturation, value = cv2.split(hsv)
#     low_sat = cv2.inRange(saturation, 0, 125)
#     bright = cv2.inRange(value, max(95, int(np.percentile(value, 35))), 255)
#     gray_bright = cv2.inRange(gray, int(np.percentile(gray, 35)), 255)
#     paper_mask = cv2.bitwise_and(cv2.bitwise_and(low_sat, bright), gray_bright)
#     paper_mask = cv2.bitwise_and(paper_mask, cv2.bitwise_not(text_mask))

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
#     paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     paper_mask = cv2.GaussianBlur(paper_mask, (0, 0), sigmaX=5)
#     return paper_mask


# def build_shadow_cleanup_mask(
#     image_bgr: np.ndarray,
#     gray: np.ndarray,
#     text_mask: np.ndarray,
#     paper_mask: np.ndarray,
# ) -> np.ndarray:
#     hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
#     _, saturation, value = cv2.split(hsv)
#     low_saturation = cv2.inRange(saturation, 0, 135)
#     not_too_dark = cv2.inRange(value, 62, 255)
#     shadow_candidate = cv2.bitwise_and(low_saturation, not_too_dark)
#     shadow_candidate = cv2.bitwise_and(shadow_candidate, cv2.bitwise_not(text_mask))

#     combined = cv2.bitwise_or(paper_mask, shadow_candidate)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
#     combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
#     combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
#     combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=8)
#     return combined


# def blend_with_mask(base: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     weight = (mask.astype("float32") / 255.0) ** 0.85
#     out = base.astype("float32") * (1.0 - weight) + cleaned.astype("float32") * weight
#     return np.clip(out, 0, 255).astype("uint8")


# def flatten_paper_variation(gray: np.ndarray, paper_mask: np.ndarray, strength: float = 0.25) -> np.ndarray:
#     paper_pixels = gray[paper_mask > 80]
#     if paper_pixels.size == 0:
#         return gray

#     target = float(np.percentile(paper_pixels, 88))
#     x = gray.astype("float32")
#     paper_weight = (paper_mask.astype("float32") / 255.0) ** 1.25
#     flattened = x + (target - x) * paper_weight * strength
#     return np.clip(flattened, 0, 255).astype("uint8")


# def remove_cast_shadows(
#     gray: np.ndarray,
#     cleanup_mask: np.ndarray,
#     text_mask: np.ndarray,
#     strength: float = 0.82,
# ) -> np.ndarray:
#     paper_pixels = gray[cleanup_mask > 80]
#     if paper_pixels.size == 0:
#         return gray

#     kernel_size = odd_kernel(max(91, min(301, int(min(gray.shape[:2]) * 0.20))))
#     background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(18, kernel_size / 4))
#     target = float(np.percentile(paper_pixels, 92))
#     deficit = np.clip(target - background.astype("float32"), 0.0, 110.0)
#     shadow_weight = np.clip(deficit / 85.0, 0.0, 1.0)
#     cleanup_weight = (cleanup_mask.astype("float32") / 255.0) ** 0.90
#     text_protection = 1.0 - (cv2.GaussianBlur(text_mask, (0, 0), sigmaX=1.0).astype("float32") / 255.0)
#     correction = deficit * shadow_weight * cleanup_weight * text_protection * strength
#     lifted = gray.astype("float32") + correction
#     return np.clip(lifted, 0, 255).astype("uint8")


# def normalize_paper_luminance(gray: np.ndarray, paper_mask: np.ndarray, strength: float = 0.55) -> np.ndarray:
#     paper_pixels = gray[paper_mask > 80]
#     if paper_pixels.size == 0:
#         return gray

#     kernel_size = background_kernel_size(gray)
#     background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(12, kernel_size / 5))
#     target = float(np.percentile(paper_pixels, 88))
#     normalized = gray.astype("float32") * target / np.maximum(background.astype("float32"), 1.0)
#     weight = (paper_mask.astype("float32") / 255.0) ** 1.10
#     out = gray.astype("float32") * (1.0 - weight * strength) + normalized * weight * strength
#     return np.clip(out, 0, 255).astype("uint8")


# def neutralize_paper_chroma(channel: np.ndarray, paper_mask: np.ndarray, strength: float = 0.35) -> np.ndarray:
#     x = channel.astype("float32")
#     paper_weight = (paper_mask.astype("float32") / 255.0) ** 1.15
#     neutralized = x * (1.0 - paper_weight * strength) + 128.0 * paper_weight * strength
#     return np.clip(neutralized, 0, 255).astype("uint8")


# def whiten_paper_regions(
#     gray: np.ndarray,
#     paper_mask: np.ndarray,
#     threshold: float = 150.0,
#     strength: float = 0.90,
# ) -> np.ndarray:
#     x = gray.astype("float32")
#     tonal_mask = np.clip((x - threshold) / (255.0 - threshold), 0.0, 1.0)
#     paper_weight = (paper_mask.astype("float32") / 255.0) * tonal_mask
#     target = 255.0 - (255.0 - x) * (1.0 - strength)
#     boosted = x * (1.0 - paper_weight) + target * paper_weight
#     return np.clip(boosted, 0, 255).astype("uint8")


# def preserve_text(enhanced: np.ndarray, source_gray: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
#     text_weight = cv2.GaussianBlur(text_mask, (0, 0), sigmaX=0.65).astype("float32") / 255.0
#     source = source_gray.astype("float32")
#     enhanced_f = enhanced.astype("float32")
#     soft_darkening = np.clip((150.0 - source) / 150.0, 0.0, 1.0)
#     text_source = source - (soft_darkening * 28.0)
#     text_source = np.minimum(text_source, enhanced_f + 6.0)
#     text_source = np.clip(text_source, 0, 210)
#     out = enhanced_f * (1.0 - text_weight) + text_source * text_weight
#     return np.clip(out, 0, 255).astype("uint8")
