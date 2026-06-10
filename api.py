"""
Egyptian ID Recognition System — FastAPI REST API
نظام التعرف على بطاقة الهوية المصرية — واجهة برمجية
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import io
import base64
import time
from PIL import Image
from typing import Optional

from core.nid_decoder import decode_national_id
from full_pipeline import run_full_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# App Setup
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="EgyptID Recognition API",
    description="REST API for Egyptian National ID card OCR & NID decoding",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # تقيّد ده لـ domain بتاعك في production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas
# ══════════════════════════════════════════════════════════════════════════════

class NIDSegments(BaseModel):
    """Breakdown of each meaningful segment inside the 14-digit NID."""
    century_digit: str = Field(..., description="الخانة الأولى (2 أو 3)")
    century_label: str = Field(..., description="القرن (١٩٠٠ أو ٢٠٠٠)")
    year_2digit:   str = Field(..., description="آخر خانتين من سنة الميلاد")
    month:         str = Field(..., description="الشهر (MM)")
    day:           str = Field(..., description="اليوم (DD)")
    governorate_code: str = Field(..., description="كود المحافظة (خانتان)")
    sequence:      str = Field(..., description="رقم التسلسل (4 خانات)")
    checksum_digit: str = Field(..., description="خانة التحقق (الخانة الأخيرة)")


class OCRToken(BaseModel):
    """Single text token returned by EasyOCR."""
    text:       str
    confidence: float = Field(..., description="درجة الثقة 0.0 → 1.0")
    confidence_pct: int = Field(..., description="درجة الثقة كنسبة مئوية 0 → 100")


class DecodedNID(BaseModel):
    """All data decoded from a valid Egyptian NID."""
    valid:       bool
    national_id: Optional[str]  = None

    # ── Core fields shown in the UI ──
    birth_date:  Optional[str]  = Field(None, description="تاريخ الميلاد DD/MM/YYYY")
    gender:      Optional[str]  = Field(None, description="الجنس: ذكر / أنثى")
    gender_en:   Optional[str]  = Field(None, description="Gender in English: male / female")
    governorate: Optional[str]  = Field(None, description="محافظة الميلاد (عربي)")
    governorate_code: Optional[str] = Field(None, description="كود المحافظة (2 digits)")
    sequence:    Optional[str]  = Field(None, description="رقم التسلسل")
    checksum_digit: Optional[str] = Field(None, description="خانة التحقق")
    century:     Optional[str]  = Field(None, description="القرن: ١٩٠٠ أو ٢٠٠٠")
    birth_year:  Optional[str]  = Field(None, description="سنة الميلاد الكاملة (4 أرقام)")
    birth_month: Optional[str]  = Field(None, description="شهر الميلاد (MM)")
    birth_day:   Optional[str]  = Field(None, description="يوم الميلاد (DD)")

    # ── Segment breakdown (for the coloured breakdown strip in the UI) ──
    segments:    Optional[NIDSegments] = Field(None, description="تفكيك الرقم القومي خانة بخانة")

    error: Optional[str] = None


class ScanResponse(BaseModel):
    success:            bool
    processing_time_ms: float

    # ── OCR raw results ──
    ocr_tokens:     list[OCRToken] = Field([], description="كل النصوص التي استخرجها OCR مع نسبة الثقة")
    ocr_text_count: int            = Field(0,  description="عدد النصوص المستخرجة")
    all_extracted_digits: Optional[str] = Field(None, description="كل الأرقام المستخرجة مجمّعة (للـ debug)")

    # ── Main results ──
    national_id: Optional[str]   = Field(None, description="الرقم القومي المستخرج (14 خانة)")
    decoded:     Optional[DecodedNID] = Field(None, description="البيانات المفككة من الرقم القومي")

    error: Optional[str] = None


class DecodeRequest(BaseModel):
    national_id: str = Field(..., min_length=14, max_length=14, example="29901011234567")


class Base64ScanRequest(BaseModel):
    image_base64:     str = Field(..., description="الصورة مُشفَّرة بـ base64")
    card_model_path:  str = Field("detect_id_card.pt")
    field_model_path: str = Field("detect_id.pt")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _enrich_decoded(raw: dict) -> DecodedNID:
    """
    Take the raw decode_national_id() dict and add every extra field
    that the Streamlit UI displays.
    """
    if not raw.get("valid"):
        return DecodedNID(valid=False, error=raw.get("error"))

    nid = raw["national_id"]

    century_digit = nid[0]
    century_label = "١٩٠٠" if century_digit == "2" else "٢٠٠٠"
    year_2digit   = nid[1:3]
    month         = nid[3:5]
    day           = nid[5:7]
    gov_code      = nid[7:9]
    sequence      = nid[9:13]
    checksum      = nid[13]
    birth_year    = ("19" if century_digit == "2" else "20") + year_2digit

    segments = NIDSegments(
        century_digit    = century_digit,
        century_label    = century_label,
        year_2digit      = year_2digit,
        month            = month,
        day              = day,
        governorate_code = gov_code,
        sequence         = sequence,
        checksum_digit   = checksum,
    )

    gender_ar = raw.get("gender", "")
    gender_en = "male" if gender_ar == "ذكر" else "female"

    return DecodedNID(
        valid            = True,
        national_id      = nid,
        birth_date       = raw.get("birth_date"),
        gender           = gender_ar,
        gender_en        = gender_en,
        governorate      = raw.get("governorate"),
        governorate_code = gov_code,
        sequence         = sequence,
        checksum_digit   = checksum,
        century          = century_label,
        birth_year       = birth_year,
        birth_month      = month,
        birth_day        = day,
        segments         = segments,
    )


def _build_scan_response(pipeline_result: dict, elapsed_ms: float) -> ScanResponse:
    from utils.helpers import normalize_digits

    # ── filter out the internal sentinel item (confidence == -1) ──
    real_ocr = [
        r for r in pipeline_result.get("ocr_results", [])
        if r.get("confidence", 0) > 0
    ]

    ocr_tokens = [
        OCRToken(
            text           = r["text"],
            confidence     = round(r["confidence"], 3),
            confidence_pct = int(r["confidence"] * 100),
        )
        for r in real_ocr
    ]

    all_digits = "".join(normalize_digits(r["text"]) for r in real_ocr) or None

    decoded_raw = pipeline_result.get("decoded")
    decoded = _enrich_decoded(decoded_raw) if decoded_raw else None

    return ScanResponse(
        success             = pipeline_result.get("success", False),
        processing_time_ms  = round(elapsed_ms, 1),
        ocr_tokens          = ocr_tokens,
        ocr_text_count      = len(ocr_tokens),
        all_extracted_digits= all_digits,
        national_id         = pipeline_result.get("national_id"),
        decoded             = decoded,
        error               = pipeline_result.get("error"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "ok",
        "service": "EgyptID Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "scan_image":   "POST /scan",
            "scan_base64":  "POST /scan/base64",
            "decode_nid":   "POST /decode",
            "decode_query": "GET  /decode/{nid}",
            "docs":         "GET  /docs",
        },
    }


@app.get("/health", tags=["Health"])
def health():
    import importlib
    checks = {}
    for lib in ["cv2", "easyocr", "PIL", "numpy", "torch", "ultralytics"]:
        try:
            mod = importlib.import_module(lib)
            checks[lib] = getattr(mod, "__version__", "ok")
        except ImportError:
            checks[lib] = "NOT INSTALLED"
    return {"status": "ok", "dependencies": checks}


# ── POST /scan ─────────────────────────────────────────────────────────────────

@app.post("/scan", response_model=ScanResponse, tags=["OCR"],
          summary="رفع صورة بطاقة هوية → OCR كامل → JSON")
async def scan_image(
    file:             UploadFile = File(...),
    card_model_path:  str        = Query("detect_id_card.pt"),
    field_model_path: str        = Query("detect_id.pt"),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are accepted")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image")

    t0 = time.perf_counter()
    result = run_full_pipeline(pil_image, card_model_path, field_model_path)
    return _build_scan_response(result, (time.perf_counter() - t0) * 1000)


# ── POST /scan/base64 ──────────────────────────────────────────────────────────

@app.post("/scan/base64", response_model=ScanResponse, tags=["OCR"],
          summary="إرسال صورة بـ base64 في JSON body")
def scan_base64(body: Base64ScanRequest):
    try:
        b64 = body.image_base64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid base64 image: {e}")

    t0 = time.perf_counter()
    result = run_full_pipeline(pil_image, body.card_model_path, body.field_model_path)
    return _build_scan_response(result, (time.perf_counter() - t0) * 1000)


# ── POST /decode ───────────────────────────────────────────────────────────────

@app.post("/decode", response_model=DecodedNID, tags=["Decoder"],
          summary="فك تشفير رقم قومي معروف بدون OCR")
def decode_nid_post(body: DecodeRequest):
    return _enrich_decoded(decode_national_id(body.national_id))


# ── GET /decode/{nid} ─────────────────────────────────────────────────────────

@app.get("/decode/{nid}", response_model=DecodedNID, tags=["Decoder"],
         summary="فك تشفير رقم قومي عبر URL")
def decode_nid_get(nid: str):
    if len(nid) != 14 or not nid.isdigit():
        raise HTTPException(status_code=422, detail="NID must be exactly 14 digits")
    return _enrich_decoded(decode_national_id(nid))


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
