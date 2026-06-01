"""
Egyptian ID Recognition System — Production UI
نظام التعرف على بطاقة الهوية المصرية
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import importlib

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EgyptID · نظام الهوية المصرية",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System & CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Cairo:wght@300;400;600;700;900&family=Syne:wght@700;800&display=swap');

  /* ── Tokens ── */
  :root {
    --bg-base:      #04080f;
    --bg-surface:   #080e1c;
    --bg-elevated:  #0c1425;
    --bg-card:      #0f1b2d;
    --border:       rgba(30, 80, 140, 0.35);
    --border-bright:rgba(56, 139, 253, 0.5);
    --accent-1:     #388bfd;
    --accent-2:     #58a6ff;
    --accent-3:     #79c0ff;
    --accent-glow:  rgba(56, 139, 253, 0.18);
    --success:      #3fb950;
    --warning:      #d29922;
    --danger:       #f85149;
    --text-primary: #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:   #484f58;
    --font-sans:    'Cairo', sans-serif;
    --font-mono:    'IBM Plex Mono', monospace;
    --font-display: 'Syne', sans-serif;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.4);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.5);
    --shadow-lg: 0 8px 40px rgba(0,0,0,0.6);
    --shadow-glow: 0 0 24px rgba(56,139,253,0.15);
  }

  /* ── Reset & Base ── */
  *, *::before, *::after { box-sizing: border-box; }
  html, body, .stApp { background: var(--bg-base) !important; }
  * { font-family: var(--font-sans) !important; }
  code, .mono { font-family: var(--font-mono) !important; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1400px !important; }
  .stDeployButton { display: none !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-surface); }
  ::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }

  /* ══════════════════════════════════════════
     TOPBAR
  ══════════════════════════════════════════ */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .topbar::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 60%, rgba(56,139,253,0.04) 100%);
    pointer-events: none;
  }
  .topbar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .topbar-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #1a3a6b 0%, #0d2140 100%);
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: var(--shadow-glow);
  }
  .topbar-name {
    font-family: var(--font-display) !important;
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
  }
  .topbar-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 1px;
  }
  .topbar-badges {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .tbadge {
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    border: 1px solid;
  }
  .tbadge-blue  { background: rgba(56,139,253,0.12); color: var(--accent-2); border-color: rgba(56,139,253,0.3); }
  .tbadge-green { background: rgba(63,185,80,0.1);   color: var(--success);  border-color: rgba(63,185,80,0.25); }

  /* ══════════════════════════════════════════
     CARDS
  ══════════════════════════════════════════ */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.2s;
  }
  .card:hover { border-color: rgba(56,139,253,0.4); }
  .card-header {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent-2);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .card-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ══════════════════════════════════════════
     DATA ROWS
  ══════════════════════════════════════════ */
  .drow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(30,80,140,0.2);
    direction: rtl;
  }
  .drow:last-child { border-bottom: none; }
  .dlabel { color: var(--text-secondary); font-size: 0.85rem; }
  .dvalue { color: var(--text-primary); font-weight: 600; font-size: 0.95rem; }
  .dvalue-mono {
    font-family: var(--font-mono) !important;
    color: var(--accent-3);
    font-size: 1.05rem;
    letter-spacing: 0.15em;
  }

  /* ══════════════════════════════════════════
     NID DISPLAY
  ══════════════════════════════════════════ */
  .nid-box {
    background: linear-gradient(135deg, #0b1e38 0%, #071530 100%);
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    font-family: var(--font-mono) !important;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--accent-2);
    letter-spacing: 0.25em;
    text-align: center;
    box-shadow: var(--shadow-glow), inset 0 1px 0 rgba(255,255,255,0.04);
    margin: 0.5rem 0 1rem;
    position: relative;
  }
  .nid-box::before {
    content: 'NATIONAL ID';
    position: absolute;
    top: 0.4rem;
    left: 0.8rem;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    color: var(--text-muted);
  }

  /* NID Segment Breakdown */
  .nid-breakdown {
    display: flex;
    gap: 4px;
    align-items: center;
    justify-content: center;
    direction: ltr;
    padding: 0.75rem;
    background: var(--bg-elevated);
    border-radius: var(--radius-sm);
    margin-top: 0.5rem;
  }
  .nid-seg {
    padding: 4px 8px;
    border-radius: 4px;
    font-family: var(--font-mono) !important;
    font-size: 1rem;
    font-weight: 600;
    border: 1px solid;
  }
  .seg-century { background: rgba(56,139,253,0.15); color: #79c0ff; border-color: rgba(56,139,253,0.3); }
  .seg-date    { background: rgba(63,185,80,0.1);   color: #7ee787; border-color: rgba(63,185,80,0.25); }
  .seg-gov     { background: rgba(210,153,34,0.12); color: #e3b341; border-color: rgba(210,153,34,0.3); }
  .seg-seq     { background: rgba(188,140,255,0.1); color: #d2a8ff; border-color: rgba(188,140,255,0.2); }
  .seg-check   { background: rgba(248,81,73,0.1);   color: #ff7b72; border-color: rgba(248,81,73,0.25); }
  .nid-legend {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.5rem;
    font-size: 0.7rem;
    color: var(--text-muted);
  }
  .leg-item { display: flex; align-items: center; gap: 4px; }
  .leg-dot { width: 8px; height: 8px; border-radius: 50%; }

  /* ══════════════════════════════════════════
     BADGES
  ══════════════════════════════════════════ */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid;
  }
  .badge-male    { background: rgba(56,139,253,0.1);  color: var(--accent-2); border-color: rgba(56,139,253,0.3); }
  .badge-female  { background: rgba(219,112,147,0.12);color: #f0a3c0;         border-color: rgba(219,112,147,0.3); }
  .badge-ok      { background: rgba(63,185,80,0.1);   color: var(--success);  border-color: rgba(63,185,80,0.25); }
  .badge-warn    { background: rgba(210,153,34,0.1);  color: var(--warning);  border-color: rgba(210,153,34,0.25); }
  .badge-fail    { background: rgba(248,81,73,0.1);   color: var(--danger);   border-color: rgba(248,81,73,0.25); }

  /* ══════════════════════════════════════════
     PIPELINE STEPS
  ══════════════════════════════════════════ */
  .pipeline {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .pstep {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.85rem;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    direction: rtl;
    transition: all 0.2s;
  }
  .pstep-idle   { background: var(--bg-elevated); border-color: var(--border); }
  .pstep-active { background: rgba(56,139,253,0.08); border-color: var(--border-bright); }
  .pstep-done   { background: rgba(63,185,80,0.06);  border-color: rgba(63,185,80,0.2); }
  .pstep-icon   { font-size: 1rem; width: 22px; text-align: center; flex-shrink: 0; }
  .pstep-label  { color: var(--text-secondary); font-size: 0.85rem; font-weight: 600; }
  .pstep-active .pstep-label { color: var(--accent-2); }
  .pstep-done   .pstep-label { color: var(--success); }

  /* ══════════════════════════════════════════
     OCR RESULTS
  ══════════════════════════════════════════ */
  .ocr-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.8rem;
    margin: 3px 0;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    direction: rtl;
  }
  .ocr-text  { color: var(--text-primary); font-size: 0.9rem; }
  .ocr-meta  { display: flex; align-items: center; gap: 0.5rem; }
  .ocr-conf  { font-family: var(--font-mono) !important; font-size: 0.75rem; color: var(--accent-2); }
  .ocr-bar-track { width: 50px; height: 3px; background: var(--bg-surface); border-radius: 2px; }
  .ocr-bar-fill  { height: 100%; border-radius: 2px; }

  /* ══════════════════════════════════════════
     UPLOAD ZONE
  ══════════════════════════════════════════ */
  [data-testid="stFileUploadDropzone"] {
    background: var(--bg-elevated) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--border-bright) !important;
  }

  /* ══════════════════════════════════════════
     SIDEBAR
  ══════════════════════════════════════════ */
  section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label { color: var(--text-secondary) !important; }
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: var(--text-primary) !important; }

  /* ══════════════════════════════════════════
     STREAMLIT OVERRIDES
  ══════════════════════════════════════════ */
  [data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
  }
  [data-testid="stMetricValue"] { color: var(--accent-2) !important; font-family: var(--font-mono) !important; }
  [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.8rem !important; }

  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    border: 1px solid var(--border);
    border-bottom: none;
    gap: 0;
    padding: 0 0.5rem;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 0 !important;
    border: none !important;
    padding: 0.65rem 1.1rem !important;
    transition: color 0.2s;
  }
  .stTabs [aria-selected="true"] {
    color: var(--accent-2) !important;
    border-bottom: 2px solid var(--accent-1) !important;
    background: transparent !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 var(--radius-md) var(--radius-md);
    padding: 1.5rem;
  }

  div.stButton > button {
    background: var(--accent-1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.25rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
    letter-spacing: 0.02em;
  }
  div.stButton > button:hover  { opacity: 0.88 !important; transform: translateY(-1px); }
  div.stButton > button:active { transform: translateY(0); }
  div.stButton > button[disabled] {
    background: var(--bg-elevated) !important;
    color: var(--text-muted) !important;
    cursor: not-allowed !important;
    transform: none !important;
  }

  div[data-testid="stTextInput"] input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    letter-spacing: 0.08em;
  }
  div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
  }

  .streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
  }

  hr { border-color: var(--border) !important; margin: 1rem 0; }

  /* ══════════════════════════════════════════
     EMPTY STATES
  ══════════════════════════════════════════ */
  .empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    color: var(--text-muted);
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
  }
  .empty-state .es-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
  .empty-state .es-title { color: var(--text-secondary); font-size: 0.95rem; font-weight: 600; margin-bottom: 0.3rem; }
  .empty-state .es-sub   { font-size: 0.8rem; }

  /* ══════════════════════════════════════════
     SYSTEM INFO TABLE
  ══════════════════════════════════════════ */
  .info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  @media (max-width: 768px) { .info-grid { grid-template-columns: 1fr; } }

  /* ══════════════════════════════════════════
     ALERTS
  ══════════════════════════════════════════ */
  .alert {
    padding: 0.75rem 1rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid;
    margin: 0.5rem 0;
  }
  .alert-warn { background: rgba(210,153,34,0.08); color: var(--warning); border-color: rgba(210,153,34,0.3); }
  .alert-fail { background: rgba(248,81,73,0.08);  color: var(--danger);  border-color: rgba(248,81,73,0.3);  }
  .alert-info { background: rgba(56,139,253,0.08); color: var(--accent-2);border-color: rgba(56,139,253,0.3); }

</style>
""", unsafe_allow_html=True)

# ── Import core modules ────────────────────────────────────────────────────────
from core.nid_decoder import decode_national_id, extract_national_id_from_text
from pipeline.full_pipeline import run_full_pipeline
from core.image_processing import preprocess_image, find_card_corners, perspective_warp
from utils.helpers import normalize_digits

# ── Topbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-icon">🪪</div>
    <div>
      <div class="topbar-name">EgyptID</div>
      <div class="topbar-sub">National ID Recognition System</div>
    </div>
  </div>
  <div class="topbar-badges">
    <span class="tbadge tbadge-blue">YOLO v8</span>
    <span class="tbadge tbadge-blue">EasyOCR</span>
    <span class="tbadge tbadge-green">● Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")
    st.markdown("---")

    st.markdown("##### 🤖 نماذج YOLO")
    card_model_path = st.text_input(
        "نموذج كشف البطاقة", value="detect_id_card.pt",
        help="مسار ملف .pt لكشف بطاقة الهوية"
    )
    field_model_path = st.text_input(
        "نموذج الحقول النصية", value="detect_id.pt",
        help="مسار ملف .pt لكشف الحقول داخل البطاقة"
    )

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(card_model_path):
            st.markdown('<span class="badge badge-ok">✓ كارت</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-warn">⚠ كارت</span>', unsafe_allow_html=True)
    with c2:
        if os.path.exists(field_model_path):
            st.markdown('<span class="badge badge-ok">✓ حقول</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-warn">⚠ حقول</span>', unsafe_allow_html=True)

    if not os.path.exists(card_model_path):
        st.markdown(
            '<div class="alert alert-info">💡 بدون .pt، النظام يعمل بـ OpenCV Edge Detection</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("##### 🔧 المعالجة")
    use_perspective  = st.toggle("تصحيح المنظور",      value=True)
    show_stages      = st.toggle("عرض مراحل المعالجة", value=True)
    ocr_conf_threshold = st.slider("حد ثقة OCR الأدنى", 0.0, 1.0, 0.3, 0.05,
                                    format="%.2f")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.8">
      <b style="color:var(--text-secondary)">Stack</b><br>
      Streamlit · YOLO v8 · EasyOCR<br>OpenCV · Pillow · PyTorch
    </div>
    """, unsafe_allow_html=True)

# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📷  التعرف على الهوية",
    "🔢  فك تشفير الرقم القومي",
    "📋  معلومات النظام",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Full Recognition Pipeline
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Upload column ──
    with col_upload:
        st.markdown("#### 📤 رفع الصورة")
        uploaded_file = st.file_uploader(
            "اختر صورة بطاقة الهوية",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="الصورة الأصلية", use_container_width=True)

            if st.button("🚀  تشغيل التحليل الكامل", type="primary", use_container_width=True):
                with st.spinner("جاري المعالجة…"):
                    stages_ph = st.empty()

                    def _render_pipeline(states: dict) -> str:
                        icons  = {"idle": "◦", "active": "◉", "done": "✓"}
                        labels = {
                            "detect":    "كشف البطاقة",
                            "warp":      "تصحيح المنظور",
                            "enhance":   "تحسين الصورة",
                            "ocr":       "استخراج النص (OCR)",
                            "decode":    "تحليل الرقم القومي",
                        }
                        rows = "".join(
                            f'<div class="pstep pstep-{states[k]}">'
                            f'<span class="pstep-icon">{icons[states[k]]}</span>'
                            f'<span class="pstep-label">{labels[k]}</span>'
                            f'</div>'
                            for k in labels
                        )
                        return f'<div class="pipeline">{rows}</div>'

                    idle_all = dict(detect="idle", warp="idle", enhance="idle", ocr="idle", decode="idle")

                    def show_pipe(**overrides):
                        s = {**idle_all, **overrides}
                        stages_ph.markdown(_render_pipeline(s), unsafe_allow_html=True)

                    show_pipe(detect="active")
                    time.sleep(0.4)
                    show_pipe(detect="done", warp="active")
                    time.sleep(0.3)
                    show_pipe(detect="done", warp="done", enhance="active")

                    result = run_full_pipeline(
                        pil_image,
                        card_model_path=card_model_path,
                        field_model_path=field_model_path,
                    )

                    show_pipe(detect="done", warp="done", enhance="done", ocr="active")
                    time.sleep(0.3)
                    show_pipe(detect="done", warp="done", enhance="done", ocr="done", decode="active")
                    time.sleep(0.2)
                    show_pipe(detect="done", warp="done", enhance="done", ocr="done", decode="done")

                    st.session_state["pipeline_result"] = result

    # ── Results column ──
    with col_result:
        st.markdown("#### 📊 النتائج")
        result = st.session_state.get("pipeline_result")

        if result:
            decoded = result.get("decoded")
            nid     = result.get("national_id")

            # ── Success path ──
            if result.get("success") and decoded and decoded.get("valid"):
                gender_badge_cls = "badge-male" if decoded["gender"] == "ذكر" else "badge-female"
                gender_icon      = "♂" if decoded["gender"] == "ذكر" else "♀"

                nid_str = decoded["national_id"]
                st.markdown(f'<div class="nid-box">{nid_str}</div>', unsafe_allow_html=True)

                # Segment breakdown
                st.markdown(f"""
                <div class="nid-breakdown">
                  <span class="nid-seg seg-century">{nid_str[0]}</span>
                  <span class="nid-seg seg-date">{nid_str[1:7]}</span>
                  <span class="nid-seg seg-gov">{nid_str[7:9]}</span>
                  <span class="nid-seg seg-seq">{nid_str[9:13]}</span>
                  <span class="nid-seg seg-check">{nid_str[13]}</span>
                </div>
                <div class="nid-legend">
                  <span class="leg-item"><span class="leg-dot" style="background:#79c0ff"></span>القرن</span>
                  <span class="leg-item"><span class="leg-dot" style="background:#7ee787"></span>تاريخ الميلاد</span>
                  <span class="leg-item"><span class="leg-dot" style="background:#e3b341"></span>المحافظة</span>
                  <span class="leg-item"><span class="leg-dot" style="background:#d2a8ff"></span>التسلسل</span>
                  <span class="leg-item"><span class="leg-dot" style="background:#ff7b72"></span>التحقق</span>
                </div>
                """, unsafe_allow_html=True)

                c_a, c_b = st.columns(2)
                with c_a: st.metric("📅 تاريخ الميلاد",   decoded["birth_date"])
                with c_b: st.metric("📍 محافظة الميلاد",  decoded["governorate"])

                st.markdown(f"""
                <div class="card">
                  <div class="card-header">📋 بيانات الهوية</div>
                  <div class="drow">
                    <span class="dlabel">الجنس</span>
                    <span class="dvalue"><span class="badge {gender_badge_cls}">{gender_icon} {decoded['gender']}</span></span>
                  </div>
                  <div class="drow">
                    <span class="dlabel">رقم التسلسل</span>
                    <span class="dvalue dvalue-mono">{decoded['sequence']}</span>
                  </div>
                  <div class="drow">
                    <span class="dlabel">خانة التحقق</span>
                    <span class="dvalue dvalue-mono">{decoded['checksum_digit']}</span>
                  </div>
                  <div class="drow">
                    <span class="dlabel">القرن</span>
                    <span class="dvalue">{'١٩٠٠' if nid_str[0]=='2' else '٢٠٠٠'}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # OCR details (collapsible)
                ocr_items = [r for r in result.get("ocr_results", [])
                             if r.get("confidence", 0) >= ocr_conf_threshold and r.get("confidence", 0) > 0]
                if ocr_items:
                    with st.expander(f"📝 نتائج OCR — {len(ocr_items)} نص", expanded=False):
                        for item in ocr_items:
                            pct = int(item["confidence"] * 100)
                            bar_color = (
                                "#3fb950" if pct >= 70 else
                                "#d29922" if pct >= 40 else "#f85149"
                            )
                            st.markdown(f"""
                            <div class="ocr-row">
                              <span class="ocr-text">{item['text']}</span>
                              <div class="ocr-meta">
                                <div class="ocr-bar-track">
                                  <div class="ocr-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                                </div>
                                <span class="ocr-conf">{pct}%</span>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

            # ── Failure path ──
            else:
                error_msg = result.get("error", "خطأ غير معروف")
                st.markdown(f"""
                <div class="card" style="border-color:rgba(248,81,73,0.35)">
                  <div class="card-header" style="color:var(--danger)">⚠ لم يُعثر على الرقم القومي</div>
                  <p style="color:#ef9a9a; font-size:0.85rem; margin:0">{error_msg}</p>
                </div>
                """, unsafe_allow_html=True)

                all_ocr = result.get("ocr_results", [])
                all_ocr = [r for r in all_ocr if r.get("confidence", 0) > 0]
                if all_ocr:
                    all_digits = "".join(normalize_digits(r["text"]) for r in all_ocr)
                    st.markdown(f"""
                    <div class="card">
                      <div class="card-header">🔢 الأرقام المستخرجة</div>
                      <div style="font-family:var(--font-mono);font-size:1rem;color:var(--accent-3);
                                  word-break:break-all;direction:ltr;padding:0.5rem;
                                  background:var(--bg-elevated);border-radius:var(--radius-sm);letter-spacing:0.12em">
                        {all_digits or '(لا توجد أرقام)'}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"📄 كل النصوص ({len(all_ocr)})", expanded=False):
                        for item in all_ocr[:25]:
                            pct = int(item["confidence"] * 100)
                            st.markdown(f"""
                            <div class="ocr-row">
                              <span class="ocr-text">{item['text']}</span>
                              <span class="ocr-conf">{pct}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="alert alert-fail">OCR لم يستخرج أي نص — تحقق من جودة الصورة وإضاءتها</div>',
                        unsafe_allow_html=True
                    )

            # ── Processing stages ──
            if show_stages and "stages" in result:
                stage_labels = {
                    "original":    "الصورة الأصلية",
                    "card_crop":   "البطاقة المقصوصة",
                    "warped":      "بعد تصحيح المنظور",
                    "preprocessed":"بعد التحسين",
                }
                with st.expander("🔬 مراحل المعالجة", expanded=False):
                    for key, label in stage_labels.items():
                        if key in result["stages"]:
                            img_rgb = cv2.cvtColor(result["stages"][key], cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, caption=label, use_container_width=True)

        else:
            st.markdown("""
            <div class="empty-state">
              <div class="es-icon">⏳</div>
              <div class="es-title">في انتظار الصورة</div>
              <div class="es-sub">ارفع صورة بطاقة الهوية وابدأ التحليل</div>
            </div>
            """, unsafe_allow_html=True)

        if not uploaded_file and not st.session_state.get("pipeline_result"):
            pass  # empty state already shown above


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual NID Decoder
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 🔢 فك تشفير الرقم القومي يدوياً")
    st.markdown(
        '<p style="color:var(--text-secondary);font-size:0.9rem;margin-bottom:1rem">'
        'أدخل الرقم القومي المكون من 14 خانة لاستخراج البيانات المضمنة فيه.'
        '</p>',
        unsafe_allow_html=True,
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        nid_input = st.text_input(
            "الرقم القومي",
            placeholder="أدخل 14 خانة…",
            max_chars=14,
            label_visibility="collapsed",
        )

        if nid_input:
            filled = len(nid_input)
            pct_done = filled / 14
            bar_col  = "#3fb950" if filled == 14 else "#388bfd" if filled >= 7 else "#d29922"
            st.markdown(f"""
            <div style="margin:-4px 0 8px">
              <div style="height:3px;background:var(--bg-elevated);border-radius:2px">
                <div style="width:{int(pct_done*100)}%;height:100%;background:{bar_col};
                            border-radius:2px;transition:width 0.2s"></div>
              </div>
              <div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;
                          font-family:var(--font-mono)">{filled}/14</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔍  فك التشفير", type="primary", use_container_width=True,
                     disabled=len(nid_input) != 14):
            st.session_state["manual_decode"] = decode_national_id(nid_input)

        st.markdown("---")
        st.markdown('<div style="color:var(--text-secondary);font-size:0.8rem;font-weight:700;'
                    'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">'
                    '💡 أمثلة</div>', unsafe_allow_html=True)

        examples = [
            ("29901011234567", "1999 · القاهرة · ذكر"),
            ("30005151234568", "2000 · الإسكندرية · أنثى"),
            ("28512241401239", "1985 · المنيا · ذكر"),
        ]
        for ex_id, ex_desc in examples:
            if st.button(f"{ex_id}  —  {ex_desc}", use_container_width=True, key=f"ex_{ex_id}"):
                st.session_state["manual_decode"] = decode_national_id(ex_id)
                st.rerun()

    with col_out:
        decoded = st.session_state.get("manual_decode")
        if decoded:
            if decoded.get("valid"):
                nid_str = decoded["national_id"]
                gender_cls  = "badge-male" if decoded["gender"] == "ذكر" else "badge-female"
                gender_icon = "♂" if decoded["gender"] == "ذكر" else "♀"

                st.markdown(f'<div class="nid-box">{nid_str}</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="nid-breakdown">
                  <span class="nid-seg seg-century">{nid_str[0]}</span>
                  <span class="nid-seg seg-date">{nid_str[1:7]}</span>
                  <span class="nid-seg seg-gov">{nid_str[7:9]}</span>
                  <span class="nid-seg seg-seq">{nid_str[9:13]}</span>
                  <span class="nid-seg seg-check">{nid_str[13]}</span>
                </div>
                <div class="nid-legend">
                  <span class="leg-item"><span class="leg-dot" style="background:#79c0ff"></span>القرن</span>
                  <span class="leg-item"><span class="leg-dot" style="background:#7ee787"></span>تاريخ الميلاد</span>
                  <span class="leg-item"><span class="leg-dot" style="background:#e3b341"></span>المحافظة</span>
                  <span class="leg-item"><span class="leg-dot" style="background:#d2a8ff"></span>التسلسل</span>
                  <span class="leg-item"><span class="leg-dot" style="background:#ff7b72"></span>التحقق</span>
                </div>
                """, unsafe_allow_html=True)

                c_a, c_b = st.columns(2)
                with c_a: st.metric("📅 تاريخ الميلاد",  decoded["birth_date"])
                with c_b: st.metric("📍 محافظة الميلاد", decoded["governorate"])

                st.markdown(f"""
                <div class="card">
                  <div class="card-header">✅ البيانات المستخرجة</div>
                  <div class="drow">
                    <span class="dlabel">الجنس</span>
                    <span class="dvalue"><span class="badge {gender_cls}">{gender_icon} {decoded['gender']}</span></span>
                  </div>
                  <div class="drow">
                    <span class="dlabel">رقم التسلسل</span>
                    <span class="dvalue dvalue-mono">{decoded['sequence']}</span>
                  </div>
                  <div class="drow">
                    <span class="dlabel">خانة التحقق</span>
                    <span class="dvalue dvalue-mono">{decoded['checksum_digit']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="card" style="border-color:rgba(248,81,73,0.35)">
                  <div class="card-header" style="color:var(--danger)">❌ رقم غير صحيح</div>
                  <p style="color:#ef9a9a;font-size:0.9rem;margin:0">{decoded.get('error','')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state" style="margin-top:1rem">
              <div class="es-icon">🔢</div>
              <div class="es-title">أدخل رقماً قومياً أو اختر مثالاً</div>
              <div class="es-sub">14 خانة تبدأ بـ 2 أو 3</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — System Info
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📋 معلومات النظام")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        # Pipeline architecture
        steps = [
            ("1", "كشف البطاقة",         "YOLO detect_id_card.pt"),
            ("2", "تصحيح المنظور",        "cv2.warpPerspective"),
            ("3", "تحسين الصورة",         "Contrast · Denoise · Sharpen"),
            ("4", "كشف الحقول (اختياري)", "YOLO detect_id.pt"),
            ("5", "استخراج النص",         "EasyOCR (AR + EN)"),
            ("6", "فك تشفير الرقم",       "Egyptian NID Decoder"),
        ]
        rows = "".join(
            f'<div class="drow">'
            f'<span class="dlabel"><span class="badge tbadge-blue" style="font-size:0.7rem;padding:1px 6px">{n}</span> {label}</span>'
            f'<span class="dvalue" style="font-size:0.8rem;color:var(--text-muted)">{tech}</span>'
            f'</div>'
            for n, label, tech in steps
        )
        st.markdown(f"""
        <div class="card">
          <div class="card-header">🏗 Pipeline المعمارية</div>
          {rows}
        </div>
        """, unsafe_allow_html=True)

        # NID structure
        segments = [
            ("الخانة 1",      "القرن",                    "2 = ١٩٠٠  |  3 = ٢٠٠٠"),
            ("الخانات 2–3",   "سنة الميلاد",              "YY"),
            ("الخانات 4–5",   "الشهر",                    "MM"),
            ("الخانات 6–7",   "اليوم",                    "DD"),
            ("الخانات 8–9",   "كود المحافظة",             "01=القاهرة … 35=جنوب سيناء"),
            ("الخانات 10–13", "رقم التسلسل",              "فردي = ذكر · زوجي = أنثى"),
            ("الخانة 14",     "خانة التحقق",               "Checksum"),
        ]
        seg_rows = "".join(
            f'<div class="drow">'
            f'<span class="dlabel">{pos}<br><small style="color:var(--text-muted)">{meaning}</small></span>'
            f'<span class="dvalue" style="font-size:0.78rem;text-align:left">{detail}</span>'
            f'</div>'
            for pos, meaning, detail in segments
        )
        st.markdown(f"""
        <div class="card">
          <div class="card-header">🗂 بنية الرقم القومي (14 خانة)</div>
          {seg_rows}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        # Dependency status
        deps = [
            ("streamlit",    "واجهة المستخدم"),
            ("cv2",          "OpenCV — معالجة الصور"),
            ("easyocr",      "EasyOCR — استخراج النص"),
            ("PIL",          "Pillow — معالجة الصور"),
            ("numpy",        "NumPy — عمليات المصفوفات"),
            ("torch",        "PyTorch — الشبكات العصبية"),
            ("ultralytics",  "Ultralytics — نماذج YOLO"),
        ]
        dep_rows = ""
        for lib, desc in deps:
            try:
                mod     = importlib.import_module(lib)
                version = getattr(mod, "__version__", "✓")
                status  = f'<span class="badge badge-ok">✓ {version}</span>'
            except ImportError:
                status = '<span class="badge badge-fail">✗ غير مثبت</span>'
            dep_rows += (
                f'<div class="drow">'
                f'<span class="dlabel">{desc}<br>'
                f'<small style="color:var(--text-muted);font-family:var(--font-mono)">{lib}</small></span>'
                f'<span class="dvalue">{status}</span>'
                f'</div>'
            )
        st.markdown(f"""
        <div class="card">
          <div class="card-header">📦 حالة المكتبات</div>
          {dep_rows}
        </div>
        """, unsafe_allow_html=True)

        # Model files status
        models = [
            ("detect_id_card.pt", "كشف بطاقة الهوية",    True),
            ("detect_id.pt",      "كشف حقول الهوية",      True),
            ("detect_odjects.pt", "كشف الكائنات (عام)",   False),
        ]
        model_rows = ""
        for mfile, mdesc, required in models:
            exists  = os.path.exists(mfile)
            badge   = (
                '<span class="badge badge-ok">✓ موجود</span>' if exists else
                f'<span class="badge {"badge-warn" if required else "badge-fail"}">{"⚠ مطلوب" if required else "✗ غير موجود"}</span>'
            )
            model_rows += (
                f'<div class="drow">'
                f'<span class="dlabel">{mdesc}<br>'
                f'<small style="color:var(--text-muted);font-family:var(--font-mono)">{mfile}</small></span>'
                f'<span class="dvalue">{badge}</span>'
                f'</div>'
            )
        st.markdown(f"""
        <div class="card">
          <div class="card-header">🤖 ملفات النماذج (.pt)</div>
          {model_rows}
          <p style="color:var(--text-muted);font-size:0.75rem;margin:0.6rem 0 0">
            ضع ملفات .pt في نفس مجلد التطبيق أو حدد المسار الكامل في الإعدادات
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:var(--text-muted);font-size:0.75rem;padding:0.5rem">
      EgyptID Recognition System · Streamlit + YOLO v8 + EasyOCR
    </div>
    """, unsafe_allow_html=True)