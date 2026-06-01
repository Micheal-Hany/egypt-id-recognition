"""OCR engine for text extraction from images."""

import numpy as np
import easyocr
from config import OCR_LANGUAGES, OCR_GPU, OCR_CONTRAST_THS, OCR_ADJUST_CONTRAST, OCR_TEXT_THRESHOLD, OCR_LOW_TEXT, OCR_DECODER, OCR_BATCH_SIZE, OCR_WORKERS


_reader = None


def get_ocr_reader():
    """
    Get or initialize the EasyOCR reader (singleton pattern).
    
    Returns:
        EasyOCR Reader instance
    """
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU, verbose=False)
    return _reader


def run_ocr_single(image: np.ndarray, conf_threshold: float = 0.2) -> list[dict]:
    """
    Run OCR on a single image array with aggressive parameters.
    
    Args:
        image: Image array (RGB)
        conf_threshold: Minimum confidence threshold
        
    Returns:
        List of OCR results with text, confidence, and bounding boxes
    """
    reader = get_ocr_reader()
    try:
        results = reader.readtext(
            image,
            contrast_ths=OCR_CONTRAST_THS,
            adjust_contrast=OCR_ADJUST_CONTRAST,
            text_threshold=OCR_TEXT_THRESHOLD,
            low_text=OCR_LOW_TEXT,
            decoder=OCR_DECODER,
            batch_size=OCR_BATCH_SIZE,
            workers=OCR_WORKERS,
        )
        return [
            {"text": t.strip(), "confidence": round(c, 3), "bbox": b}
            for (b, t, c) in results
            if t.strip() and c >= conf_threshold
        ]
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return []


def _bbox_left_x(item: dict) -> float:
    """Get the leftmost X coordinate of a bounding box."""
    bbox = item.get("bbox")
    if bbox is None:
        return 0.0
    try:
        return min(pt[0] for pt in bbox)
    except Exception:
        return 0.0


def _bbox_top_y(item: dict) -> float:
    """Get the topmost Y coordinate of a bounding box."""
    bbox = item.get("bbox")
    if bbox is None:
        return 0.0
    try:
        return min(pt[1] for pt in bbox)
    except Exception:
        return 0.0


def _sort_rtl(items: list[dict]) -> list[dict]:
    """
    Sort OCR tokens for Arabic right-to-left text.
    Groups tokens into rows by Y proximity, then within each row
    sorts right-to-left (descending X).
    
    Args:
        items: List of OCR result items
        
    Returns:
        Spatially sorted list of items
    """
    if not items:
        return items

    # Estimate line height as median bbox height
    heights = []
    for item in items:
        bbox = item.get("bbox")
        if bbox:
            try:
                h = max(pt[1] for pt in bbox) - min(pt[1] for pt in bbox)
                heights.append(h)
            except Exception:
                pass
    line_height = float(np.median(heights)) if heights else 20.0
    row_tolerance = max(line_height * 0.6, 10.0)

    # Group into rows
    sorted_by_y = sorted(items, key=_bbox_top_y)
    rows: list[list[dict]] = []
    for item in sorted_by_y:
        y = _bbox_top_y(item)
        placed = False
        for row in rows:
            row_y = _bbox_top_y(row[0])
            if abs(y - row_y) <= row_tolerance:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    # Within each row sort right-to-left (descending left_x)
    result = []
    for row in rows:
        row.sort(key=_bbox_left_x, reverse=True)
        result.extend(row)

    return result


def run_ocr(image: np.ndarray) -> list[dict]:
    """
    Multi-strategy OCR: try all image variants, collect results with
    spatial ordering preserved (RTL), deduplicate by text keeping highest
    confidence, and return combined results.
    
    Args:
        image: Image array (RGB or RGBA)
        
    Returns:
        List of deduplicated OCR results with variant sequences attached
    """
    import cv2
    from .image_processing import make_ocr_variants
    
    if image.shape[2] == 3:
        bgr = image
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    variants = make_ocr_variants(bgr)

    # best_items: text → item with highest confidence (bbox preserved)
    best_items: dict[str, dict] = {}
    # Also keep per-variant spatially-ordered lists for NID extraction
    variant_ordered: list[list[dict]] = []

    for name, variant_img in variants:
        ocr_out = run_ocr_single(variant_img)
        ordered = _sort_rtl(ocr_out)
        variant_ordered.append(ordered)
        for item in ocr_out:
            t = item["text"]
            c = item["confidence"]
            if t not in best_items or c > best_items[t]["confidence"]:
                best_items[t] = item

    # Return the deduplicated list
    result_list = sorted(best_items.values(), key=lambda x: -x["confidence"])

    # Store ordered variants as a special marker item for extract_national_id_from_text
    result_list.append({
        "text": "",
        "confidence": -1.0,
        "bbox": None,
        "_variant_sequences": variant_ordered,
    })

    return result_list
