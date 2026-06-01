"""
Configuration and constants for Egyptian ID Recognition System.
"""

# Governorate codes to Arabic names mapping
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

# Century digit to year prefix mapping
CENTURY_MAP = {"2": "19", "3": "20"}

# Arabic-Indic (٠١٢٣٤٥٦٧٨٩) to Latin digits translation table
AR_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Image processing parameters
MIN_OCR_WIDTH = 1200
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# OCR parameters
OCR_LANGUAGES = ['ar', 'en']
OCR_GPU = False
OCR_CONTRAST_THS = 0.05
OCR_ADJUST_CONTRAST = 0.8
OCR_TEXT_THRESHOLD = 0.4
OCR_LOW_TEXT = 0.3
OCR_DECODER = 'greedy'
OCR_BATCH_SIZE = 1
OCR_WORKERS = 0

# Card detection parameters
CARD_MIN_WIDTH = 100
CARD_MIN_HEIGHT = 50

# Perspective correction parameters
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 200
CONTOUR_APPROX_EPSILON = 0.02

# Expected NID format
NID_LENGTH = 14
NID_VALID_FIRST_DIGITS = "23"
