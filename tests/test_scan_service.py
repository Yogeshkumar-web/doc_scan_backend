import pytest
import numpy as np
import cv2
from app.services.scan_service import find_document_contour, order_points, four_point_transform
from app.services.enhance_service import enhance_document
from app.services.scanner.enhancement import (
    build_auto_candidates,
    build_text_mask,
    compute_stroke_metrics,
    enhance_clean_grayscale,
    enhance_document as enhance_document_result,
    enhance_print_clean,
    remove_large_mask_components,
)

@pytest.fixture
def synthetic_document_image():
    """Generates a synthetic image of a document (white rectangle) on a dark background."""
    # 800x600 dark grey background
    img = np.ones((800, 600, 3), dtype="uint8") * 30
    
    # Define a clean rectangle representing the document: corners at (100, 150), (500, 100), (520, 700), (80, 720)
    # Let's draw a rotated white polygon
    pts = np.array([[100, 150], [500, 100], [520, 700], [80, 720]], dtype="int32")
    cv2.fillPoly(img, [pts], (240, 240, 240))
    
    return img

@pytest.fixture
def synthetic_noisy_image():
    """Generates a noisy image with no clear document structure."""
    return np.random.randint(0, 255, (400, 400, 3), dtype="uint8")

def test_find_document_contour_success(synthetic_document_image):
    contour = find_document_contour(synthetic_document_image)
    assert contour is not None
    assert len(contour) == 4
    # Check that points are roughly in the same ballpark as the drawn rectangle
    # Since points could be returned in any order, we sort them first or check membership
    for pt in contour:
        assert len(pt) == 2
        assert 0 <= pt[0] <= 600
        assert 0 <= pt[1] <= 800

def test_find_document_contour_failure(synthetic_noisy_image):
    contour = find_document_contour(synthetic_noisy_image)
    assert contour is None

def test_order_points():
    # Points in random order
    pts = np.array([[100, 100], [400, 100], [400, 300], [100, 300]], dtype="float32")
    shuffled_pts = np.array([[400, 300], [100, 100], [100, 300], [400, 100]], dtype="float32")
    
    ordered = order_points(shuffled_pts)
    
    # Order should be: top-left, top-right, bottom-right, bottom-left
    np.testing.assert_array_almost_equal(ordered[0], [100, 100])
    np.testing.assert_array_almost_equal(ordered[1], [400, 100])
    np.testing.assert_array_almost_equal(ordered[2], [400, 300])
    np.testing.assert_array_almost_equal(ordered[3], [100, 300])

def test_four_point_transform(synthetic_document_image):
    # Retrieve the contour we detected
    contour = find_document_contour(synthetic_document_image)
    assert contour is not None
    
    # Warping should crop the document out
    warped = four_point_transform(synthetic_document_image, contour)
    assert warped is not None
    assert warped.shape[0] > 0
    assert warped.shape[1] > 0
    # The output should be smaller than original but match aspect ratio of document roughly
    assert warped.shape[0] < synthetic_document_image.shape[0]

@pytest.mark.parametrize("mode", ["auto", "print", "color", "gray", "bw", "soft"])
def test_enhance_document_modes(mode, synthetic_document_image):
    enhanced = enhance_document(synthetic_document_image, mode)
    assert enhanced is not None
    assert enhanced.shape == synthetic_document_image.shape
    assert enhanced.dtype == np.uint8

def test_print_mode_whitens_uneven_background():
    img = np.ones((700, 500, 3), dtype="uint8") * 215
    gradient = np.linspace(80, 0, img.shape[1], dtype="uint8")
    img = np.clip(img - gradient[None, :, None], 0, 255).astype("uint8")
    cv2.putText(img, "Sample document text", (45, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2)
    cv2.putText(img, "with uneven lighting", (45, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (45, 45, 45), 2)

    auto_result = enhance_document_result(img, "auto")
    color_result = enhance_document_result(img, "color")

    assert auto_result.metrics.background_whiteness > color_result.metrics.background_whiteness
    assert auto_result.metrics.text_contrast >= 0.35

def test_auto_mode_can_fallback_from_scary_binary_output():
    img = np.ones((700, 500, 3), dtype="uint8") * 215
    cv2.rectangle(img, (0, 0), (160, 700), (120, 120, 120), -1)
    cv2.putText(img, "Document text", (45, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2)
    cv2.putText(img, "with notes", (45, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (45, 45, 45), 2)

    result = enhance_document_result(img, "auto")

    assert result.metrics.selected_enhancement in {"clean_grayscale", "clean_color", "print_clean"}
    assert result.metrics.background_whiteness >= 0.65
    assert result.metrics.text_contrast >= 0.25


def test_auto_mode_generates_three_named_candidates():
    img = np.ones((620, 440, 3), dtype="uint8") * 220
    cv2.rectangle(img, (0, 0), (440, 620), (235, 235, 225), 8)
    cv2.putText(img, "Invoice", (55, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (35, 35, 35), 2)
    cv2.putText(img, "Amount 1250", (55, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (45, 45, 45), 2)

    candidates = build_auto_candidates(img)

    assert [candidate.name for candidate in candidates] == ["clean_color", "clean_grayscale", "print_clean"]
    assert all(candidate.image.shape == img.shape for candidate in candidates)
    assert all(np.isfinite(candidate.score) for candidate in candidates)
    assert all(candidate.metrics.selected_enhancement == candidate.name for candidate in candidates)


def test_auto_mode_selects_highest_scored_candidate():
    img = np.ones((620, 440, 3), dtype="uint8") * 210
    gradient = np.linspace(55, 0, img.shape[1], dtype="uint8")
    img = np.clip(img - gradient[None, :, None], 0, 255).astype("uint8")
    cv2.putText(img, "Document", (55, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2)
    cv2.putText(img, "Clean scan", (55, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (45, 45, 45), 2)

    candidates = build_auto_candidates(img)
    expected = max(candidates, key=lambda candidate: candidate.score).name
    result = enhance_document_result(img, "auto")

    assert result.metrics.selected_enhancement == expected


def test_clean_grayscale_lifts_paper_background_without_losing_text():
    img = np.ones((500, 360, 3), dtype="uint8") * 220
    gradient = np.linspace(85, 0, img.shape[1], dtype="uint8")
    img = np.clip(img - gradient[None, :, None], 0, 255).astype("uint8")
    cv2.putText(img, "Shadow text", (45, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.putText(img, "Line two", (45, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (45, 45, 45), 2)

    source_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhance_clean_grayscale(img), cv2.COLOR_BGR2GRAY)
    text_mask = build_text_mask(source_gray) > 0
    paper_region = np.zeros(source_gray.shape, dtype=bool)
    paper_region[40:460, 20:340] = True
    paper_region &= ~text_mask

    assert float(np.mean(enhanced_gray[paper_region])) > float(np.mean(source_gray[paper_region])) + 25
    assert float(np.percentile(enhanced_gray[paper_region], 90)) > 235
    assert int(np.min(enhanced_gray[130:260, 35:270])) <= 45


def test_print_clean_avoids_binary_threshold_when_shadow_is_not_safe():
    img = np.ones((500, 360, 3), dtype="uint8") * 220
    cv2.rectangle(img, (0, 0), (135, 500), (105, 105, 105), -1)
    cv2.putText(img, "Shadow text", (45, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.putText(img, "Line two", (45, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (45, 45, 45), 2)

    enhanced_gray = cv2.cvtColor(enhance_print_clean(img), cv2.COLOR_BGR2GRAY)

    assert len(np.unique(enhanced_gray)) > 2


def test_clean_grayscale_does_not_create_ink_spread():
    img = np.ones((520, 380, 3), dtype="uint8") * 218
    gradient = np.linspace(55, 0, img.shape[1], dtype="uint8")
    img = np.clip(img - gradient[None, :, None], 0, 255).astype("uint8")
    cv2.putText(img, "Thin legal text", (44, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (42, 42, 42), 1)
    cv2.putText(img, "No ink spread", (44, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (48, 48, 48), 1)
    cv2.rectangle(img, (42, 270), (330, 390), (70, 70, 70), 1)

    source_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhance_clean_grayscale(img), cv2.COLOR_BGR2GRAY)
    source_ink_ratio = float(np.mean(source_gray < 110))
    enhanced_ink_ratio = float(np.mean(enhanced_gray < 110))

    assert enhanced_ink_ratio <= source_ink_ratio + 0.035
    assert compute_stroke_metrics(enhanced_gray)["median_stroke_width"] <= 3.2


def test_auto_prefers_soft_grayscale_over_harsh_binary_for_shadowed_page():
    img = np.ones((620, 440, 3), dtype="uint8") * 218
    cv2.rectangle(img, (245, 280), (440, 620), (135, 135, 135), -1)
    cv2.putText(img, "Court index", (58, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (38, 38, 38), 2)
    cv2.putText(img, "Application", (58, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), 1)
    cv2.rectangle(img, (55, 270), (390, 560), (70, 70, 70), 1)

    result = enhance_document_result(img, "auto")

    assert result.metrics.selected_enhancement in {"clean_grayscale", "clean_color"}


def test_clean_grayscale_removes_broad_shadow_without_thickening_text():
    img = np.ones((620, 440, 3), dtype="uint8") * 225
    cv2.rectangle(img, (250, 280), (440, 610), (120, 120, 120), -1)
    cv2.circle(img, (290, 355), 36, (105, 105, 105), -1)
    cv2.putText(img, "Court index", (60, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (38, 38, 38), 2)
    cv2.putText(img, "Application", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (48, 48, 48), 1)
    cv2.rectangle(img, (55, 270), (392, 560), (70, 70, 70), 1)

    source_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhance_clean_grayscale(img), cv2.COLOR_BGR2GRAY)
    shadow_region = np.zeros(source_gray.shape, dtype=bool)
    shadow_region[300:560, 260:420] = True

    assert float(np.mean(enhanced_gray[shadow_region])) > float(np.mean(source_gray[shadow_region])) + 90
    assert float(np.percentile(enhanced_gray[shadow_region], 10)) > 225
    assert compute_stroke_metrics(enhanced_gray)["median_stroke_width"] <= 3.2


def test_remove_large_mask_components_keeps_small_components_only():
    mask = np.zeros((100, 100), dtype="uint8")
    cv2.rectangle(mask, (5, 5), (12, 12), 255, -1)
    cv2.rectangle(mask, (40, 40), (95, 95), 255, -1)

    cleaned = remove_large_mask_components(mask, max_area_ratio=0.05)

    assert int(cleaned[8, 8]) == 255
    assert int(cleaned[60, 60]) == 0
