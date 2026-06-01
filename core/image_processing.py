"""Image preprocessing and perspective correction utilities."""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from config import MIN_OCR_WIDTH, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2, CONTOUR_APPROX_EPSILON


def make_ocr_variants(bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Generate multiple preprocessed versions of the image.
    EasyOCR will be run on each; the best result wins.
    
    Args:
        bgr: Image in BGR format (OpenCV standard)
        
    Returns:
        List of (variant_name, image_array) tuples
    """
    variants = []
    h, w = bgr.shape[:2]

    # Ensure minimum size for OCR (upscale if too small)
    if w < MIN_OCR_WIDTH:
        scale = MIN_OCR_WIDTH / w
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Raw RGB
    variants.append(("raw", rgb))

    # 2. Upscaled x2
    up2 = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants.append(("upscale_x2", up2))

    # 3. Contrast + Sharpness (PIL)
    pil = Image.fromarray(rgb)
    pil = ImageEnhance.Contrast(pil).enhance(2.0)
    pil = ImageEnhance.Sharpness(pil).enhance(2.5)
    pil = ImageEnhance.Brightness(pil).enhance(1.1)
    variants.append(("enhanced", np.array(pil)))

    # 4. CLAHE on grayscale
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    cl = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)))

    # 5. Grayscale only
    variants.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)))

    # 6. Denoised
    denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    variants.append(("denoised", cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)))

    # 7. Bilateral filter (preserves edges)
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    variants.append(("bilateral", cv2.cvtColor(bilateral, cv2.COLOR_GRAY2RGB)))

    # 8. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants.append(("adaptive_thresh", cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)))

    # 9. Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)))

    # 10. Inverted (handles dark backgrounds)
    inv = cv2.bitwise_not(gray)
    variants.append(("inverted", cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)))

    return variants


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing for display purposes.
    
    Args:
        image: Image in BGR format
        
    Returns:
        Preprocessed image in BGR format
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.8)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return denoised


def find_card_corners(image: np.ndarray):
    """
    Find the corners of an ID card in the image using edge detection.
    
    Args:
        image: Image in BGR format
        
    Returns:
        Array of 4 corner points or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, CONTOUR_APPROX_EPSILON * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points in clockwise manner starting from top-left.
    
    Args:
        pts: Array of 4 points
        
    Returns:
        Ordered array of 4 points
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to straighten an ID card.
    
    Args:
        image: Image in BGR format
        corners: Array of 4 corner points
        
    Returns:
        Perspective-corrected image
    """
    rect = order_points(corners)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))
