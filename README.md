# EgyptID — Egyptian National ID Recognition System
### نظام التعرف على بطاقة الهوية المصرية

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=flat-square)
![EasyOCR](https://img.shields.io/badge/EasyOCR-AR%20%2B%20EN-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A production-grade pipeline for detecting, extracting, and decoding Egyptian National ID cards from images — combining YOLO object detection, multi-strategy EasyOCR, and a rule-based NID decoder.  
Comes with both a **Streamlit UI** and a **FastAPI REST API** for backend integration.

</div>

---

## ✨ Features

- **Automatic card detection** via YOLOv8 with OpenCV edge-based fallback
- **Perspective correction** to straighten tilted or angled card photos
- **10-variant OCR pipeline** with Arabic-Indic numeral normalization and RTL digit-group reversal
- **Robust NID extraction** — handles OCR misreads, split tokens, and mixed Arabic/Latin digits
- **Full NID decoder** — birth date, governorate, gender, sequence number, and checksum
- **Manual decoder tab** for direct 14-digit input with live visual breakdown
- **Dark production UI** with real-time pipeline progress, confidence bars, and processing stage previews
- **REST API** (FastAPI) — upload an image or send base64, receive structured JSON with all decoded fields

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- (Optional) NVIDIA GPU for faster YOLO and EasyOCR inference
- Linux: `sudo apt install libgl1` required for OpenCV headless

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/egypt-id-recognition.git
cd egypt-id-recognition

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Launch the Streamlit UI
streamlit run APP.py

# 4b. OR launch the REST API
python api.py
```

- Streamlit UI → **http://localhost:8501**
- REST API → **http://localhost:8000**
- Swagger docs → **http://localhost:8000/docs**

### One-line scripts

```bash
# Linux / macOS
chmod +x run.sh && ./run.sh

# Windows
run.bat
```

---

## 📁 Project Structure

```
egypt-id-recognition/
├── APP.py                  # Streamlit application (UI layer)
├── api.py                  # FastAPI REST API
├── full_pipeline.py        # Core pipeline orchestration
├── config.py               # Constants and configuration
├── debug_ocr.py            # Step-by-step OCR diagnostic tool
├── requirements.txt        # Python dependencies
├── run.sh                  # Launch script (Linux/macOS)
├── run.bat                 # Launch script (Windows)
│
├── core/
│   ├── models.py           # YOLO model loading
│   ├── image_processing.py # Preprocessing variants & perspective warp
│   ├── ocr_engine.py       # EasyOCR engine & RTL sorting
│   └── nid_decoder.py      # NID extraction & decoding
│
├── utils/
│   └── helpers.py          # Digit normalization utilities
│
├── detect_id_card.pt       # ← Place your YOLO card-detection model here
├── detect_id.pt            # ← Place your YOLO field-detection model here
└── detect_odjects.pt       # ← Optional general-object detection model
```

> **No `.pt` files?** The system falls back automatically to OpenCV Canny edge detection + full-image OCR. Detection accuracy will be lower but the pipeline remains functional.

---

## 🌐 REST API

The `api.py` file exposes the full pipeline as a FastAPI service — no UI needed. Ideal for mobile apps, backend services, or any system that needs to process ID cards programmatically.

### Start the API server

```bash
# Install extra deps (if not already installed)
pip install fastapi uvicorn[standard] python-multipart

python api.py
# → http://localhost:8000
# → http://localhost:8000/docs  (interactive Swagger UI)
```

---

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check & endpoint map |
| `GET`  | `/health` | Dependency version check |
| `POST` | `/scan` | Upload image file → full pipeline → JSON |
| `POST` | `/scan/base64` | Same but image as base64 in JSON body |
| `POST` | `/decode` | Decode a known 14-digit NID string (no OCR) |
| `GET`  | `/decode/{nid}` | Same via URL path |

---

### POST `/scan` — Upload an image

```bash
curl -X POST http://localhost:8000/scan \
  -F "file=@id_card.jpg"
```

**Full JSON response:**

```json
{
  "success": true,
  "processing_time_ms": 1842.3,

  "national_id": "29901011234567",

  "ocr_text_count": 18,
  "ocr_tokens": [
    { "text": "محمد أحمد علي",   "confidence": 0.923, "confidence_pct": 92 },
    { "text": "29901011234567",   "confidence": 0.871, "confidence_pct": 87 },
    { "text": "القاهرة",          "confidence": 0.654, "confidence_pct": 65 }
  ],
  "all_extracted_digits": "2990101123456701011234567",

  "decoded": {
    "valid": true,
    "national_id":  "29901011234567",

    "birth_date":   "01/01/1999",
    "birth_year":   "1999",
    "birth_month":  "01",
    "birth_day":    "01",

    "gender":       "ذكر",
    "gender_en":    "male",

    "governorate":       "القاهرة",
    "governorate_code":  "01",

    "century":       "١٩٠٠",
    "sequence":      "1234",
    "checksum_digit": "7",

    "segments": {
      "century_digit":    "2",
      "century_label":    "١٩٠٠",
      "year_2digit":      "99",
      "month":            "01",
      "day":              "01",
      "governorate_code": "01",
      "sequence":         "1234",
      "checksum_digit":   "7"
    },

    "error": null
  },

  "error": null
}
```

> `ocr_tokens` contains **all raw text** detected on the card (names, address, etc.) with per-token confidence scores.

---

### POST `/scan/base64` — Base64 image (mobile-friendly)

```bash
curl -X POST http://localhost:8000/scan/base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."}'
```

Same response shape as `/scan`.

---

### POST `/decode` — Decode NID without OCR

```bash
curl -X POST http://localhost:8000/decode \
  -H "Content-Type: application/json" \
  -d '{"national_id": "29901011234567"}'
```

### GET `/decode/{nid}` — Decode via URL

```bash
curl http://localhost:8000/decode/29901011234567
```

**Response (both decode endpoints):**

```json
{
  "valid":        true,
  "national_id":  "29901011234567",
  "birth_date":   "01/01/1999",
  "birth_year":   "1999",
  "birth_month":  "01",
  "birth_day":    "01",
  "gender":       "ذكر",
  "gender_en":    "male",
  "governorate":  "القاهرة",
  "governorate_code": "01",
  "century":      "١٩٠٠",
  "sequence":     "1234",
  "checksum_digit": "7",
  "segments": { ... },
  "error": null
}
```

---

### Flutter / Dart example

```dart
// Upload image file
final request = http.MultipartRequest(
  'POST',
  Uri.parse('http://YOUR_SERVER_IP:8000/scan'),
);
request.files.add(await http.MultipartFile.fromPath('file', imagePath));
final streamedResponse = await request.send();
final body = jsonDecode(await streamedResponse.stream.bytesToString());

final nid       = body['national_id'];
final decoded   = body['decoded'];
final gender    = decoded['gender'];        // ذكر / أنثى
final genderEn  = decoded['gender_en'];     // male / female
final gov       = decoded['governorate'];   // القاهرة
final birthDate = decoded['birth_date'];    // 01/01/1999
final birthYear = decoded['birth_year'];    // 1999
final century   = decoded['century'];       // ١٩٠٠ or ٢٠٠٠
final ocrTexts  = (body['ocr_tokens'] as List)
                    .map((t) => t['text'] as String)
                    .toList();              // raw card text (names, etc.)
```

---

### Production deployment notes

```bash
# Single worker is important — EasyOCR reader is a global singleton
gunicorn -w 1 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
```

- Replace `allow_origins=["*"]` in `api.py` with your actual domain before going live.
- The YOLO `.pt` model files must be present at the paths passed as query parameters (default: same directory as `api.py`).

---

## 🗂 Egyptian National ID Structure

```
  [C] [YY MM DD] [GG] [SSSS] [K]
   │    │  │  │   │     │     └─ Checksum digit
   │    │  │  │   │     └─────── 4-digit serial  (odd = male, even = female)
   │    │  │  │   └───────────── Governorate code (01 = Cairo … 35 = South Sinai)
   │    │  │  └───────────────── Birth day   (DD)
   │    │  └──────────────────── Birth month (MM)
   │    └─────────────────────── Birth year  (YY, last two digits)
   └──────────────────────────── Century     (2 = 1900s, 3 = 2000s)
```

**Example:** `30101282600418`  
→ Born 28 Jan 2001 · Governorate 26 (Sohag) · Male · Serial 0041 · Checksum 8

---

## 🔧 OCR Pipeline Details

The system applies up to **10 image preprocessing variants** per image and runs EasyOCR on each:

| Variant | Technique |
|---------|-----------|
| `raw` | Original RGB |
| `upscale_x2` | 2× bicubic upscaling |
| `enhanced` | Contrast + Sharpness + Brightness (PIL) |
| `clahe` | Contrast Limited Adaptive Histogram Equalization |
| `gray` | Grayscale |
| `denoised` | Fast non-local means denoising |
| `bilateral` | Bilateral filter (edge-preserving) |
| `adaptive_thresh` | Adaptive Gaussian threshold |
| `otsu` | Otsu's global threshold |
| `inverted` | Bitwise inversion (for dark backgrounds) |

**NID extraction strategies (in priority order):**

1. **RTL group reversal** — reverses space-separated digit groups within a single token to reconstruct Arabic RTL order (e.g. `١٨ ٠٠٤ ٢٦ ٢٨ ٠١ ٠١ ٣` → `30101282600418`)
2. **Spatial RTL sort** — groups tokens into rows, sorts each row right-to-left by X coordinate
3. **Per-token search** — normalized Arabic-Indic digits searched individually
4. **Concatenated search** — all tokens joined then searched
5. **Sliding window** — 14-character windows with first-digit misread correction (`5/8/9 → 3`, `7/4 → 2`)
6. **Relaxed validation** — date check disabled as last resort

---

## 🐛 Debugging

Use `debug_ocr.py` to diagnose recognition failures step by step:

```bash
python debug_ocr.py path/to/your_id_image.jpg
```

This runs the full preprocessing and OCR chain, saves variant images (`debug_variant_*.jpg`), and prints normalized digit strings and NID search results for each variant.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.28 | Web UI |
| `fastapi` | ≥0.111 | REST API framework |
| `uvicorn[standard]` | ≥0.29 | ASGI server |
| `python-multipart` | ≥0.0.9 | File upload support for FastAPI |
| `ultralytics` | ≥8.0 | YOLOv8 inference |
| `easyocr` | ≥1.7 | Arabic + English OCR |
| `opencv-python-headless` | ≥4.8 | Image processing |
| `Pillow` | ≥10.0 | Image enhancement |
| `torch` | ≥2.0 | PyTorch (YOLO + OCR backend) |
| `numpy` | ≥1.24 | Array operations |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

All runtime settings are available in the sidebar (Streamlit UI) or as query parameters (API):

| Setting | Default | Description |
|---------|---------|-------------|
| Card model path | `detect_id_card.pt` | Path to YOLO card detection model |
| Field model path | `detect_id.pt` | Path to YOLO field detection model |
| Perspective correction | `on` | Apply warpPerspective after detection |
| Show processing stages | `on` | Display intermediate pipeline images (UI only) |
| OCR confidence threshold | `0.30` | Minimum confidence to show OCR results |

---

## 🔒 Privacy Notice

This application processes images locally. No data is sent to external servers. Images are held in memory only for the duration of a single request and are not persisted to disk.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with Streamlit · FastAPI · YOLO v8 · EasyOCR · OpenCV
</div>
