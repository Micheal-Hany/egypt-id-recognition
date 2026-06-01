"""
Debug Tool — يشخص مشكلة OCR خطوة بخطوة
شغّله: python debug_ocr.py path/to/your_image.jpg
"""

import sys
import cv2
import numpy as np
from PIL import Image
import traceback
import re
import os

# Import from refactored modules
from core.models import load_yolo_model
from core.image_processing import make_ocr_variants
from core.ocr_engine import run_ocr_single, _sort_rtl
from core.nid_decoder import decode_national_id
from utils.helpers import normalize_digits, normalize_digits_rtl, fix_first_digit_misreads


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
card_model_path = "detect_id_card.pt"
card_crop = img_bgr.copy()

if not os.path.exists(card_model_path):
    warn(f"ملف {card_model_path} مش موجود — هيستخدم الصورة كاملة")
else:
    try:
        model = load_yolo_model(card_model_path)
        if model:
            results = model(img_bgr, verbose=False)
            boxes = results[0].boxes
            if boxes and len(boxes) > 0:
                box = boxes[0].xyxy[0].cpu().numpy().astype(int)
                x1,y1,x2,y2 = box
                card_crop = img_bgr[y1:y2, x1:x2]
                ok(f"تم اكتشاف البطاقة: bbox=[{x1},{y1},{x2},{y2}], crop size={card_crop.shape[1]}x{card_crop.shape[0]}")
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

variants = {}
for name, variant_img in make_ocr_variants(card_crop):
    variants[name] = variant_img
    cv2.imwrite(f"debug_variant_{name}.jpg", cv2.cvtColor(variant_img, cv2.COLOR_RGB2BGR))

ok(f"تم تجهيز {len(variants)} نسخة من الصورة")
ok("تم حفظ debug_variant_*.jpg — افتحهم وشوف أي واحد أوضح")

# ── 4. EasyOCR on each variant ────────────────────────────────────────────────
step("4. تشغيل EasyOCR على كل نسخة")

try:
    ok("تم تحميل EasyOCR بنجاح")
except Exception as e:
    err(f"فشل تحميل EasyOCR: {e}")
    sys.exit(1)

best_nid = None
best_variant = None
all_results = {}


def _find_nid(digit_str):
    """Search for 14-digit NID, trying common OCR misreads on the first digit."""
    for variant in fix_first_digit_misreads(digit_str):
        m = re.search(r'[23]\d{13}', variant)
        if m:
            nid = m.group()
            month, day = int(nid[3:5]), int(nid[5:7])
            if 1 <= month <= 12 and 1 <= day <= 31:
                return nid
    # Relax date check as last resort
    for variant in fix_first_digit_misreads(digit_str):
        m = re.search(r'[23]\d{13}', variant)
        if m:
            return m.group()
    return None

for name, img_variant in variants.items():
    try:
        raw_results = run_ocr_single(img_variant)
        texts = [r["text"] for r in raw_results]
        confs = [r["confidence"] for r in raw_results]
        all_results[name] = {"texts": texts, "confs": confs}

        # Strategy 0: try RTL group reversal on each token individually (best for Arabic)
        found_nid = None
        for r in sorted(raw_results, key=lambda x: -x["confidence"]):
            found_nid = _find_nid(normalize_digits_rtl(r["text"]))
            if found_nid:
                break
            found_nid = _find_nid(normalize_digits(r["text"]))
            if found_nid:
                break

        # Strategy 1: RTL spatial order across tokens (requires reconstructing OCR results)
        if not found_nid:
            # Fallback to original order
            digits_only = "".join(normalize_digits(t) for t in texts)
            found_nid = _find_nid(digits_only)

        status = f"{len(texts)} نص"
        if found_nid and not best_nid:
            best_nid = found_nid
            best_variant = name
            status += f" 🎯 رقم قومي: {best_nid}"
        elif texts:
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
    digits = "".join(normalize_digits(t) for t in data["texts"])
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
            digits = "".join(normalize_digits(t) for t in data["texts"])
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