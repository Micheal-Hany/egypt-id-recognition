#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Egyptian ID Recognition System — Setup & Run Script
# ═══════════════════════════════════════════════════════════════

echo "🪪  Egyptian ID Recognition System"
echo "═══════════════════════════════════"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 غير موجود. قم بتثبيته أولاً."
    exit 1
fi

echo "✅ Python: $(python3 --version)"

# Check/install pip dependencies
echo ""
echo "📦 تثبيت المتطلبات..."
pip install -r requirements.txt -q

# System dependency (libgl1 for OpenCV on Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 نظام Linux — التحقق من libgl1..."
    dpkg -l libgl1 &> /dev/null || sudo apt-get install -y libgl1 -q
fi

echo ""
echo "🚀 تشغيل التطبيق..."
echo "   افتح المتصفح على: http://localhost:8501"
echo "   اضغط Ctrl+C للإيقاف"
echo ""

streamlit run APP.py \
    --server.port 8501 \
    --server.headless false \
    --theme.base dark \
    --theme.primaryColor "#42a5f5" \
    --theme.backgroundColor "#0a0e1a" \
    --theme.secondaryBackgroundColor "#0d1529" \
    --theme.textColor "#e8f4fd"
python debug_ocr.py photo_5821385810031349567_y.jpg







python debug_ocr.py photo_5821385810031349567_y.jpg

──────────────────────────────────────────────────
  1. تحميل الصورة
──────────────────────────────────────────────────
  ✅ تم تحميل الصورة: 960x582 px

──────────────────────────────────────────────────
  2. كشف البطاقة بـ YOLO
──────────────────────────────────────────────────
  ✅ تم اكتشاف البطاقة: bbox=[0,19,879,576], crop size=879x557
  ✅ تم حفظ debug_card_crop.jpg
  ℹ️  حجم الصورة اللي هتتعمل عليها OCR: 879x557

──────────────────────────────────────────────────
  3. تجهيز الصور للـ OCR (variants)
──────────────────────────────────────────────────
  ✅ تم تجهيز 6 نسخة من الصورة
  ✅ تم حفظ debug_variant_*.jpg — افتحهم وشوف أي واحد أوضح

──────────────────────────────────────────────────
  4. تشغيل EasyOCR على كل نسخة
──────────────────────────────────────────────────
  ✅ تم تحميل EasyOCR بنجاح
C:\Users\Micheal\anaconda3\Lib\site-packages\torch\utils\data\dataloader.py:752: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
  ℹ️  [gray                ] 14 نص → "زكنفضمزالح | 9{ | السخقية..."
  ℹ️  [upscale_x2          ] 16 نص → "جمهوزتيذف ص الح بينا | السخقية | بطاقة ...."
  ℹ️  [enhanced            ] 14 نص → "جمهو تزفد | الشخصية | بطاقة..."
  ℹ️  [adaptive_thresh     ] 15 نص → "جهعوذتذفط زالج بيغما |  السخفية | بطاقة..."
  ℹ️  [clahe               ] 11 نص → "الشخقية | بطاقة | صابر عبدالله ابراهيم..."

──────────────────────────────────────────────────
  5. تحليل الأرقام المستخرجة
──────────────────────────────────────────────────
  ℹ️  [raw_rgb             ] أرقام (normalized): 5110200818004262801014454209
  ℹ️    → أرقام طويلة: ['5110200818004262801014454209']
  ℹ️  [gray                ] أرقام (normalized): 91102008180042628014454209
  ℹ️    → أرقام طويلة: ['91102008180042628014454209']
  ℹ️  [upscale_x2          ] أرقام (normalized): 783180042628014454209
  ℹ️    → أرقام طويلة: ['783180042628014454209']
  ℹ️  [enhanced            ] أرقام (normalized): 7020004026281014454209
  ℹ️    → أرقام طويلة: ['7020004026281014454209']
  ℹ️  [adaptive_thresh     ] أرقام (normalized): 848180042628144542097
  ℹ️    → أرقام طويلة: ['848180042628144542097']
  ℹ️  [clahe               ] أرقام (normalized): 7607644454209
  ℹ️    → أرقام طويلة: ['7607644454209']

──────────────────────────────────────────────────
  6. ملخص النتائج
──────────────────────────────────────────────────
  ✅ ✅ تم العثور على الرقم القومي: 26280101811020
  ✅    أفضل نسخة: raw_rgb
  ⚠️     تحقق فشل: تاريخ الميلاد في الرقم القومي غير صحيح

══════════════════════════════════════════════════
  تم الانتهاء من الـ Debug
══════════════════════════════════════════════════