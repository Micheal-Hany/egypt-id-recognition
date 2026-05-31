# EgyptID — Egyptian National ID Recognition System
### نظام التعرف على بطاقة الهوية المصرية

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=flat-square)
![EasyOCR](https://img.shields.io/badge/EasyOCR-AR%20%2B%20EN-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A production-grade pipeline for detecting, extracting, and decoding Egyptian National ID cards from images — combining YOLO object detection, multi-strategy EasyOCR, and a rule-based NID decoder.

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

# 4. Launch the app
streamlit run APP.py
```

Open your browser at **http://localhost:8501**

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
├── utils.py                # Core pipeline: YOLO · OCR · Decoder
├── debug_ocr.py            # Step-by-step OCR diagnostic tool
├── requirements.txt        # Python dependencies
├── run.sh                  # Launch script (Linux/macOS)
├── run.bat                 # Launch script (Windows)
│
├── detect_id_card.pt       # ← Place your YOLO card-detection model here
├── detect_id.pt            # ← Place your YOLO field-detection model here
└── detect_odjects.pt       # ← Optional general-object detection model
```

> **No `.pt` files?** The system falls back automatically to OpenCV Canny edge detection + full-image OCR. Detection accuracy will be lower but the pipeline remains functional.

---

## 🤖 YOLO Models

Place trained `.pt` files in the project root directory:

| File | Purpose | Required |
|------|---------|----------|
| `detect_id_card.pt` | Detects and crops the ID card from a scene image | Recommended |
| `detect_id.pt` | Detects individual text fields within the cropped card | Optional |
| `detect_odjects.pt` | General object detection | Optional |

You can supply custom model paths at runtime via the sidebar settings.

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

All runtime settings are available in the sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| Card model path | `detect_id_card.pt` | Path to YOLO card detection model |
| Field model path | `detect_id.pt` | Path to YOLO field detection model |
| Perspective correction | `on` | Apply warpPerspective after detection |
| Show processing stages | `on` | Display intermediate pipeline images |
| OCR confidence threshold | `0.30` | Minimum confidence to show OCR results |

---

## 🔒 Privacy Notice

This application processes images locally. No data is sent to external servers. Images are held in memory only for the duration of a single session and are not persisted to disk.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with Streamlit · YOLO v8 · EasyOCR · OpenCV
</div>s