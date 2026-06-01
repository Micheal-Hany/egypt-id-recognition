"""Main recognition pipeline orchestration."""

import cv2
import numpy as np
from PIL import Image
from core.models import load_yolo_model
from core.image_processing import find_card_corners, perspective_warp, preprocess_image
from core.ocr_engine import run_ocr
from core.nid_decoder import extract_national_id_from_text, decode_national_id
from utils.helpers import normalize_digits
from config import CARD_MIN_WIDTH, CARD_MIN_HEIGHT


def run_full_pipeline(
    image_input,
    card_model_path: str = "detect_id_card.pt",
    field_model_path: str = "detect_id.pt",
) -> dict:
    """
    Execute the complete ID recognition pipeline: detection → OCR → decoding.
    
    Args:
        image_input: PIL Image, numpy array, or file path
        card_model_path: Path to YOLO card detection model
        field_model_path: Path to YOLO field detection model (optional)
        
    Returns:
        Dictionary with pipeline results, stages, OCR results, and decoded NID
    """
    result = {
        "success": False,
        "stages": {},
        "ocr_results": [],
        "national_id": None,
        "decoded": None,
        "error": None,
    }

    # ── Load image ──
    if isinstance(image_input, np.ndarray):
        image = image_input.copy()
    elif isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        try:
            pil = Image.open(image_input).convert("RGB")
            image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            result["error"] = f"خطأ في تحميل الصورة: {e}"
            return result

    result["stages"]["original"] = image.copy()

    # ── Stage 1: Card Detection ──
    card_model = load_yolo_model(card_model_path)
    card_crop = image.copy()

    if card_model:
        try:
            detections = card_model(image, verbose=False)
            for det in detections:
                if det.boxes and len(det.boxes) > 0:
                    box = det.boxes[0].xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    # Sanity check: crop must be reasonably sized
                    if (x2 - x1) > CARD_MIN_WIDTH and (y2 - y1) > CARD_MIN_HEIGHT:
                        card_crop = image[y1:y2, x1:x2]
                    break
        except Exception as e:
            print(f"[WARN] Card model inference failed: {e}")

    # Fallback: try perspective correction on full image
    if card_crop is image:
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
