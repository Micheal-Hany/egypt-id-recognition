"""
Debug Tool — يشخص مشكلة OCR خطوة بخطوة
شغّله: python debug_ocr.py path/to/your_image.jpg
"""

import sys
import cv2
import numpy as np
from PIL import Image
import traceback

def step(msg):
    print(f"\n{'─'*50}")
    print(f"  {msg}")
    print('─'*50)

def ok(msg):   print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def err(msg):  print(f"  ❌ {msg}")
def info(msg): print(f"  ℹ️  {msg}")

# ── 1. Image loading ──────────────────────────────────────────────────────────
step("1. تحميل الصورة")
image_path = sys.argv[1] if len(sys.argv) > 1 else None
if not image_path:
    err("استخدم: python debug_ocr.py path/to/image.jpg")
    sys.exit(1)

try:
    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    ok(f"تم تحميل الصورة: {img_bgr.shape[1]}x{img_bgr.shape[0]} px")
except Exception as e:
    err(f"فشل تحميل الصورة: {e}")
    sys.exit(1)

# ── 2. YOLO card detection ─────────────────────────────────────────────────────
step("2. كشف البطاقة بـ YOLO")
import os
card_model_path = "detect_id_card.pt"
card_crop = img_bgr.copy()

if not os.path.exists(card_model_path):
    warn(f"ملف {card_model_path} مش موجود — هيستخدم الصورة كاملة")
else:
    try:
        from ultralytics import YOLO
        model = YOLO(card_model_path)
        results = model(img_bgr, verbose=False)
        boxes = results[0].boxes
        if boxes and len(boxes) > 0:
            box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1,y1,x2,y2 = box
            card_crop = img_bgr[y1:y2, x1:x2]
            ok(f"تم اكتشاف البطاقة: bbox=[{x1},{y1},{x2},{y2}], crop size={card_crop.shape[1]}x{card_crop.shape[0]}")
            # Save debug crop
            cv2.imwrite("debug_card_crop.jpg", card_crop)
            ok("تم حفظ debug_card_crop.jpg")
        else:
            warn("YOLO مش شايف بطاقة — هيستخدم الصورة كاملة")
    except Exception as e:
        warn(f"خطأ في YOLO: {e}")
        warn("هيستخدم الصورة كاملة")

info(f"حجم الصورة اللي هتتعمل عليها OCR: {card_crop.shape[1]}x{card_crop.shape[0]}")

# ── 3. Preprocessing variants ─────────────────────────────────────────────────
step("3. تجهيز الصور للـ OCR (variants)")

def make_variants(bgr):
    variants = {}
    # Raw RGB
    variants["raw_rgb"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variants["gray"]    = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Upscaled x2
    h,w = bgr.shape[:2]
    up  = cv2.resize(bgr, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    variants["upscale_x2"] = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    # Contrast enhanced
    from PIL import Image as PILImage, ImageEnhance
    pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Contrast(pil).enhance(2.0)
    pil = ImageEnhance.Sharpness(pil).enhance(2.5)
    variants["enhanced"] = np.array(pil)
    # Adaptive threshold on gray
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    variants["adaptive_thresh"] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    variants["clahe"] = cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)

    return variants

variants = make_variants(card_crop)
ok(f"تم تجهيز {len(variants)} نسخة من الصورة")
for name, v in variants.items():
    cv2.imwrite(f"debug_variant_{name}.jpg", cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
ok("تم حفظ debug_variant_*.jpg — افتحهم وشوف أي واحد أوضح")

# ── 4. EasyOCR on each variant ────────────────────────────────────────────────
step("4. تشغيل EasyOCR على كل نسخة")
import easyocr
import re

try:
    reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
    ok("تم تحميل EasyOCR بنجاح")
except Exception as e:
    err(f"فشل تحميل EasyOCR: {e}")
    sys.exit(1)

best_nid = None
best_variant = None
all_results = {}

# Arabic-Indic → Latin digit map (must happen BEFORE regex search)
AR_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def _normalize(text):
    """Convert Arabic-Indic numerals to Latin and strip non-digit chars."""
    return re.sub(r'\D', '', text.translate(AR_DIGIT_MAP))

def _normalize_rtl(text):
    """
    For tokens where digit groups are separated by spaces in RTL visual order
    (e.g. '١٨ ٠٠٤ ٢٦ ٢٨ ٠١ ٠١ ٣'), reverse the groups before stripping spaces.
    '١٨ ٠٠٤ ٢٦ ٢٨ ٠١ ٠١ ٣' → reversed groups → '30101282600418'
    """
    translated = text.translate(AR_DIGIT_MAP)
    groups = translated.split()
    if len(groups) > 1:
        return re.sub(r'\D', '', "".join(reversed(groups)))
    return re.sub(r'\D', '', translated)

def _sort_rtl(raw_results):
    """
    Sort EasyOCR results for Arabic RTL text.
    raw_results: list of (bbox, text, conf) tuples from readtext().
    Groups into rows by Y proximity, then sorts each row right-to-left.
    Returns list of text strings in RTL reading order.
    """
    if not raw_results:
        return []
    items = [{"bbox": b, "text": t, "conf": c} for b, t, c in raw_results]
    heights = []
    for item in items:
        try:
            h = max(pt[1] for pt in item["bbox"]) - min(pt[1] for pt in item["bbox"])
            heights.append(h)
        except Exception:
            pass
    line_h = float(np.median(heights)) if heights else 20.0
    tol = max(line_h * 0.6, 10.0)

    def top_y(item):
        try: return min(pt[1] for pt in item["bbox"])
        except: return 0.0
    def left_x(item):
        try: return min(pt[0] for pt in item["bbox"])
        except: return 0.0

    sorted_y = sorted(items, key=top_y)
    rows = []
    for item in sorted_y:
        y = top_y(item)
        placed = False
        for row in rows:
            if abs(y - top_y(row[0])) <= tol:
                row.append(item); placed = True; break
        if not placed:
            rows.append([item])
    result = []
    for row in rows:
        row.sort(key=left_x, reverse=True)  # RTL: rightmost first
        result.extend(item["text"] for item in row)
    return result

def _find_nid(digit_str):
    """
    Search for 14-digit NID, also trying common OCR misreads on the first digit.
    3 is frequently misread as 5, 8, or 9 — and 2 as 7, 4, or 6.
    """
    first_digit_fixes = {'5':'3','8':'3','9':'3','7':'2','4':'2','6':'2'}
    variants_to_try = [digit_str]
    if digit_str and digit_str[0] in first_digit_fixes:
        variants_to_try.append(first_digit_fixes[digit_str[0]] + digit_str[1:])
    for v in variants_to_try:
        m = re.search(r'[23]\d{13}', v)
        if m:
            nid = m.group()
            month, day = int(nid[3:5]), int(nid[5:7])
            if 1 <= month <= 12 and 1 <= day <= 31:
                return nid
    # Relax date check as last resort
    for v in variants_to_try:
        m = re.search(r'[23]\d{13}', v)
        if m:
            return m.group()
    return None

for name, img_variant in variants.items():
    try:
        raw_results = reader.readtext(
            img_variant,
            contrast_ths=0.05,
            adjust_contrast=0.8,
            text_threshold=0.4,
            low_text=0.3,
            decoder='greedy',
            batch_size=1,
        )
        texts = [r[1] for r in raw_results]
        confs = [r[2] for r in raw_results]
        all_results[name] = {"texts": texts, "confs": confs, "raw": raw_results}

        # Strategy 0: try RTL group reversal on each token individually (best for Arabic)
        # e.g. token '١٨ ٠٠٤ ٢٦ ٢٨ ٠١ ٠١ ٣' → '30101282600418'
        found_nid = None
        for r in sorted(raw_results, key=lambda x: -x[2]):  # highest confidence first
            found_nid = _find_nid(_normalize_rtl(r[1]))
            if found_nid:
                break
            found_nid = _find_nid(_normalize(r[1]))
            if found_nid:
                break

        # Strategy 1: RTL spatial order across tokens
        if not found_nid:
            rtl_texts = _sort_rtl(raw_results)
            rtl_digits = "".join(_normalize_rtl(t) for t in rtl_texts)
            found_nid = _find_nid(rtl_digits)

        # Strategy 2: fallback to original order
        if not found_nid:
            digits_only = "".join(_normalize(t) for t in texts)
            found_nid = _find_nid(digits_only)

        class _Match:
            def group(self): return found_nid
        nid_match = _Match() if found_nid else None

        status = f"{len(texts)} نص"
        if nid_match and not best_nid:
            best_nid = nid_match.group()
            best_variant = name
            status += f" 🎯 رقم قومي: {best_nid}"
        elif texts:
            # Show first 3 texts
            preview = " | ".join(texts[:3])[:60]
            status += f' → "{preview}..."'
        else:
            status += " (فاضي)"

        info(f"[{name:20s}] {status}")

    except Exception as e:
        warn(f"[{name:20s}] خطأ: {e}")

# ── 5. Digit extraction analysis ─────────────────────────────────────────────
step("5. تحليل الأرقام المستخرجة")
for name, data in all_results.items():
    # Normalize Arabic-Indic before analysis
    digits = "".join(_normalize(t) for t in data["texts"])
    if digits:
        info(f"[{name:20s}] أرقام (normalized): {digits[:50]}")
        long_nums = re.findall(r'\d{6,}', digits)
        if long_nums:
            info(f"  → أرقام طويلة: {long_nums}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
step("6. ملخص النتائج")
if best_nid:
    ok(f"✅ تم العثور على الرقم القومي: {best_nid}")
    ok(f"   أفضل نسخة: {best_variant}")
    # Decode
    from utils import decode_national_id
    decoded = decode_national_id(best_nid)
    if decoded["valid"]:
        ok(f"   تاريخ الميلاد: {decoded['birth_date']}")
        ok(f"   الجنس: {decoded['gender']}")
        ok(f"   المحافظة: {decoded['governorate']}")
    else:
        warn(f"   تحقق فشل: {decoded['error']}")
else:
    err("لم يتم العثور على الرقم القومي في أي نسخة!")
    print()
    print("  الأرقام الموجودة في كل النسخ (بعد التطبيع):")
    for name, data in all_results.items():
        if data["texts"]:
            digits = "".join(_normalize(t) for t in data["texts"])
            if digits:
                print(f"  [{name}]: {digits[:80]}")
    print()
    warn("جرب الحلول التالية:")
    warn("1. الصورة محتاجة تكون أعلى وضوح (min 300 DPI أو 1000px عرض)")
    warn("2. تأكد إن الرقم القومي ظاهر واضح بدون انعكاس أو ظل")
    warn("3. جرب تصوير البطاقة مباشرة من فوق بإضاءة جيدة")

print(f"\n{'═'*50}")
print("  تم الانتهاء من الـ Debug")
print(f"{'═'*50}")