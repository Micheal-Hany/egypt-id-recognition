"""
Egyptian ID Recognition System - Core Utilities v2
Robust pipeline with multiple OCR strategies and fallbacks.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import re
import os

# ─── YOLO Model Loader ────────────────────────────────────────────────────────

def load_yolo_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        print(f"[WARN] Could not load model {model_path}: {e}")
        return None


# ─── Image Preprocessing Variants ─────────────────────────────────────────────

def make_ocr_variants(bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Generate multiple preprocessed versions of the image.
    EasyOCR will be run on each; the best result wins.
    """
    variants = []
    h, w = bgr.shape[:2]

    # Ensure minimum size for OCR (upscale if too small)
    min_w = 1200
    if w < min_w:
        scale = min_w / w
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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
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
    """Standard preprocessing for display purposes."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.8)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return denoised


# ─── Perspective Correction ────────────────────────────────────────────────────

def find_card_corners(image: np.ndarray):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    rect = order_points(corners)
    (tl, tr, br, bl) = rect
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    maxW    = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH    = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


# ─── OCR Engine ───────────────────────────────────────────────────────────────

_reader = None

def get_ocr_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
    return _reader


def run_ocr_single(image: np.ndarray, conf_threshold: float = 0.2) -> list[dict]:
    """Run OCR with aggressive parameters on a single image array."""
    reader = get_ocr_reader()
    try:
        results = reader.readtext(
            image,
            contrast_ths=0.05,
            adjust_contrast=0.8,
            text_threshold=0.4,
            low_text=0.3,
            decoder='greedy',
            batch_size=1,
            workers=0,
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
    """Return the leftmost X coordinate of a bbox (list of 4 points)."""
    bbox = item.get("bbox")
    if bbox is None:
        return 0.0
    try:
        return min(pt[0] for pt in bbox)
    except Exception:
        return 0.0


def _bbox_top_y(item: dict) -> float:
    """Return the topmost Y coordinate of a bbox."""
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
    sorts right-to-left (descending X).  This ensures the NID digits,
    which are printed RTL, are concatenated in the correct order.
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
    """
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

    # Attach ordered variant lists for downstream extraction
    # Return the deduplicated list, but also attach the per-variant ordered sequences
    result_list = sorted(best_items.values(), key=lambda x: -x["confidence"])

    # Store ordered variants as a special marker item for extract_national_id_from_text
    result_list.append({
        "text": "",
        "confidence": -1.0,
        "bbox": None,
        "_variant_sequences": variant_ordered,
    })

    return result_list


# ─── National ID Decoder ──────────────────────────────────────────────────────

GOVERNORATES = {
    "01": "القاهرة",       "02": "الإسكندرية",  "03": "بور سعيد",
    "04": "السويس",        "11": "دمياط",        "12": "الدقهلية",
    "13": "الشرقية",       "14": "القليوبية",    "15": "كفر الشيخ",
    "16": "الغربية",       "17": "المنوفية",     "18": "البحيرة",
    "19": "الإسماعيلية",   "21": "الجيزة",       "22": "بني سويف",
    "23": "الفيوم",        "24": "المنيا",       "25": "أسيوط",
    "26": "سوهاج",         "27": "قنا",          "28": "أسوان",
    "29": "الأقصر",        "31": "البحر الأحمر", "32": "الوادي الجديد",
    "33": "مطروح",         "34": "شمال سيناء",   "35": "جنوب سيناء",
    "88": "خارج الجمهورية",
}
CENTURY_MAP = {"2": "19", "3": "20"}

# Arabic-Indic (٠١٢٣٤٥٦٧٨٩) → Latin digits
AR_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic numerals to Latin and strip non-digit chars."""
    return re.sub(r'\D', '', text.translate(AR_DIGIT_MAP))


def normalize_digits_rtl(text: str) -> str:
    """
    For tokens where EasyOCR returned Arabic digit groups with spaces
    (e.g. '١٨ ٠٠٤ ٢٦ ٢٨ ٠١ ٠١ ٣'), the groups are in RTL visual order —
    rightmost group appears first in the string.
    Reversing the groups gives LTR order suitable for regex matching.

    '١٨ ٠٠٤ ٢٦ ٢٨ ٠١ ٠١ ٣'
     → groups reversed → '٣ ٠١ ٠١ ٢٨ ٢٦ ٠٠٤ ١٨'
     → normalized      → '30101282600418'  ✓
    """
    translated = text.translate(AR_DIGIT_MAP)
    groups = translated.split()
    if len(groups) > 1:
        return re.sub(r'\D', '', "".join(reversed(groups)))
    return re.sub(r'\D', '', translated)


def decode_national_id(nid: str) -> dict:
    nid = normalize_digits(nid)
    if len(nid) != 14:
        return {"valid": False, "error": f"يجب أن يكون الرقم 14 خانة، تم إدخال {len(nid)} خانة"}

    century_digit = nid[0]
    if century_digit not in CENTURY_MAP:
        return {"valid": False, "error": "الخانة الأولى غير صحيحة (يجب أن تكون 2 أو 3)"}

    year  = CENTURY_MAP[century_digit] + nid[1:3]
    month = nid[3:5]
    day   = nid[5:7]
    gov   = nid[7:9]
    seq   = nid[9:13]
    gender_digit = int(nid[12])
    checksum     = nid[13]

    try:
        if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
            raise ValueError
    except ValueError:
        return {"valid": False, "error": "تاريخ الميلاد في الرقم القومي غير صحيح"}

    return {
        "valid":       True,
        "national_id": nid,
        "birth_date":  f"{day}/{month}/{year}",
        "gender":      "ذكر" if gender_digit % 2 != 0 else "أنثى",
        "governorate": GOVERNORATES.get(gov, f"محافظة غير معروفة ({gov})"),
        "sequence":    seq,
        "checksum_digit": checksum,
    }


def _fix_first_digit_misreads(digit_str: str) -> list[str]:
    """
    Return variants of digit_str with common OCR misreads on the first digit corrected.
    OCR often confuses 2↔5, 3↔8, 3↔9, 2↔7 on Arabic ID cards.
    Only the first digit matters for NID validity (must be 2 or 3).
    """
    if not digit_str:
        return [digit_str]
    variants = [digit_str]
    # Map of misread first-digit → correct first-digit
    first_digit_fixes = {
        '5': '3', '8': '3', '9': '3',   # 3 misread as 5, 8, or 9
        '7': '2', '4': '2',              # 2 misread as 7 or 4
        '6': '2',                        # 2 misread as 6 (less common)
    }
    if digit_str[0] in first_digit_fixes:
        variants.append(first_digit_fixes[digit_str[0]] + digit_str[1:])
    return variants


def extract_national_id_from_text(ocr_results: list[dict]) -> str | None:
    """
    Robust NID extraction with RTL spatial ordering as the primary strategy.

    Strategy 0 (RTL spatial): use per-variant spatially-sorted token sequences
                               (tokens ordered right-to-left per row) so Arabic
                               NID digits are concatenated in the correct order.
    Strategy A: search each token individually after normalizing Arabic digits.
    Strategy B: concatenate all normalized tokens and search.
    Strategy C: sliding window with first-digit misread correction.
    Strategy D: relax date validation as last resort.
    """
    def is_valid_candidate(c: str) -> bool:
        if not re.fullmatch(r'[23]\d{13}', c):
            return False
        month = int(c[3:5])
        day   = int(c[5:7])
        return 1 <= month <= 12 and 1 <= day <= 31

    def search_in_digit_string(digit_str: str, strict: bool = True) -> str | None:
        for variant in _fix_first_digit_misreads(digit_str):
            m = re.search(r'[23]\d{13}', variant)
            if m:
                if strict and is_valid_candidate(m.group()):
                    return m.group()
                elif not strict and re.fullmatch(r'[23]\d{13}', m.group()):
                    return m.group()
        return None

    # ── Strategy -1: per-token RTL group reversal (highest priority) ─────────
    # Handles the common case where EasyOCR returns a single token like
    # '١٨ ٠٠٤ ٢٦ ٢٨ ٠١ ٠١ ٣' — groups are in RTL visual order, reversing them
    # gives the correct LTR digit string '30101282600418'.
    real_results = [r for r in ocr_results if r.get("confidence", 0) >= 0]

    # Sort by confidence descending — try high-confidence tokens first
    for item in sorted(real_results, key=lambda x: -x.get("confidence", 0)):
        rtl_norm = normalize_digits_rtl(item["text"])
        found = search_in_digit_string(rtl_norm, strict=True)
        if found:
            return found
        # Also try the non-reversed form for tokens that are already LTR
        ltr_norm = normalize_digits(item["text"])
        if ltr_norm != rtl_norm:
            found = search_in_digit_string(ltr_norm, strict=True)
            if found:
                return found

    # ── Strategy 0: RTL-ordered per-variant sequences ──────────────────────────
    # The _variant_sequences marker item is appended by run_ocr()
    variant_sequences = None
    for item in ocr_results:
        if item.get("confidence") == -1.0 and "_variant_sequences" in item:
            variant_sequences = item["_variant_sequences"]
            break

    if variant_sequences:
        for ordered_seq in variant_sequences:
            # Try both normalizers on RTL-ordered token sequence
            for norm_fn in (normalize_digits_rtl, normalize_digits):
                rtl_digits = "".join(norm_fn(item["text"]) for item in ordered_seq)
                found = search_in_digit_string(rtl_digits, strict=True)
                if found:
                    return found
                pure = "".join(ch for ch in rtl_digits if ch.isdigit())
                for i in range(len(pure) - 13):
                    found = search_in_digit_string(pure[i:i + 14], strict=True)
                    if found:
                        return found

    # ── Strategy A: per-token search ──────────────────────────────────────────
    normalized_texts = [normalize_digits(r["text"]) for r in real_results]

    for digits in normalized_texts:
        found = search_in_digit_string(digits, strict=True)
        if found:
            return found

    # ── Strategy B: concatenated (LTR fallback) ───────────────────────────────
    all_digits = "".join(normalized_texts)
    found = search_in_digit_string(all_digits, strict=True)
    if found:
        return found

    # ── Strategy C: sliding window ────────────────────────────────────────────
    concat = "".join(ch for ch in all_digits if ch.isdigit())
    for i in range(len(concat) - 13):
        found = search_in_digit_string(concat[i:i + 14], strict=True)
        if found:
            return found

    # ── Strategy D: relax date validation ─────────────────────────────────────
    if variant_sequences:
        for ordered_seq in variant_sequences:
            rtl_digits = "".join(normalize_digits(item["text"]) for item in ordered_seq)
            pure = "".join(ch for ch in rtl_digits if ch.isdigit())
            for i in range(len(pure) - 13):
                found = search_in_digit_string(pure[i:i + 14], strict=False)
                if found:
                    return found

    for i in range(len(concat) - 13):
        found = search_in_digit_string(concat[i:i + 14], strict=False)
        if found:
            return found

    return None


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    image_input,
    card_model_path: str = "detect_id_card.pt",
    field_model_path: str = "detect_id.pt",
) -> dict:
    result = {
        "success":     False,
        "stages":      {},
        "ocr_results": [],
        "national_id": None,
        "decoded":     None,
        "error":       None,
    }

    # ── Load image ──
    if isinstance(image_input, np.ndarray):
        image = image_input.copy()
    elif isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        try:
            pil   = Image.open(image_input).convert("RGB")
            image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            result["error"] = f"خطأ في تحميل الصورة: {e}"
            return result

    result["stages"]["original"] = image.copy()

    # ── Stage 1: Card Detection ──
    card_model = load_yolo_model(card_model_path)
    card_crop  = image.copy()

    if card_model:
        try:
            detections = card_model(image, verbose=False)
            for det in detections:
                if det.boxes and len(det.boxes) > 0:
                    box = det.boxes[0].xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    # Sanity check: crop must be reasonably sized
                    if (x2 - x1) > 100 and (y2 - y1) > 50:
                        card_crop = image[y1:y2, x1:x2]
                    break
        except Exception as e:
            print(f"[WARN] Card model inference failed: {e}")

    # Fallback: try perspective correction on full image
    if card_crop is image:  # no YOLO crop happened
        corners = find_card_corners(image)
        if corners is not None:
            try:
                card_crop = perspective_warp(image, corners)
            except Exception:
                pass

    result["stages"]["card_crop"] = card_crop.copy()

    # ── Stage 2: Perspective Warp (post YOLO) ──
    corners = find_card_corners(card_crop)
    if corners is not None:
        try:
            card_crop = perspective_warp(card_crop, corners)
        except Exception:
            pass
    result["stages"]["warped"] = card_crop.copy()

    # ── Stage 3: Preprocess for display ──
    try:
        preprocessed = preprocess_image(card_crop)
    except Exception:
        preprocessed = card_crop.copy()
    result["stages"]["preprocessed"] = preprocessed.copy()

    # ── Stage 4: OCR (multi-strategy) ──
    # Use preprocessed card for OCR
    result["ocr_results"] = run_ocr(preprocessed)

    # If still empty, try the raw card crop
    if not result["ocr_results"]:
        result["ocr_results"] = run_ocr(card_crop)

    # If STILL empty, try the full original image
    if not result["ocr_results"]:
        result["ocr_results"] = run_ocr(image)

    # ── Stage 5: Extract & Decode NID ──
    nid = extract_national_id_from_text(result["ocr_results"])
    result["national_id"] = nid

    if nid:
        result["decoded"] = decode_national_id(nid)
        result["success"] = result["decoded"].get("valid", False)
    else:
        # Build a helpful debug message
        all_digits = "".join(
            normalize_digits(r["text"]) for r in result["ocr_results"]
        )
        if all_digits:
            result["error"] = f"OCR وجد أرقاماً ({all_digits[:40]}...) لكن لم يتطابق مع نمط الرقم القومي (14 خانة تبدأ بـ 2 أو 3)"
        elif result["ocr_results"]:
            all_text = " | ".join(r["text"] for r in result["ocr_results"][:5])
            result["error"] = f"OCR وجد نصاً بدون أرقام: {all_text[:80]}"
        else:
            result["error"] = "OCR لم يستخرج أي نص — تأكد من جودة الصورة وإضاءتها"

    return result