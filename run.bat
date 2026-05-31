@echo off
echo 🪪  Egyptian ID Recognition System
echo ===================================
echo.

echo 📦 تثبيت المتطلبات...
pip install -r requirements.txt

echo.
echo 🚀 تشغيل التطبيق...
echo    افتح المتصفح على: http://localhost:8501
echo    اضغط Ctrl+C للإيقاف
echo.

streamlit run APP.py ^
    --server.port 8501 ^
    --theme.base dark ^
    --theme.primaryColor "#42a5f5" ^
    --theme.backgroundColor "#0a0e1a" ^
    --theme.secondaryBackgroundColor "#0d1529" ^
    --theme.textColor "#e8f4fd"

pause
