"""
X-Ray Fracture Detection Frontend (YOLOv11 + FastAPI)
OrthoScan — Clinical Diagnostic Imaging UI
"""
 
import os
import base64
from io import BytesIO
from datetime import datetime
 
import requests
import streamlit as st
from PIL import Image
 
# --------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
 
st.set_page_config(
    page_title="OrthoScan — Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# --------------------------------------------------------------------------
# Load External CSS
# --------------------------------------------------------------------------
CSS_PATH = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
 
def img_to_data_uri(img: Image.Image) -> str:
    """Encode a PIL image as a base64 data URI for embedding in raw HTML."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"
 
 
# --------------------------------------------------------------------------
# Header Section
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-container">
        <div class="eyebrow">Diagnostic Imaging &middot; AI-Assisted Detection</div>
        <h1>X-Ray Fracture Detection</h1>
        <p class="subtitle">Upload an X-ray image and run inference against a YOLOv11
        object-detection model to localize suspected fractures.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
st.markdown(
    f"""
    <div class="meta-strip">
        <div class="meta-item">
            <div class="meta-label">Model</div>
            <div class="meta-value">YOLOv11 &middot; Object Detection</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Accepted Formats</div>
            <div class="meta-value">JPG &middot; JPEG &middot; PNG</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Session</div>
            <div class="meta-value">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Status</div>
            <div class="meta-value status-ready">● Ready</div>
        </div>
    </div>
    <hr class="divider" />
    """,
    unsafe_allow_html=True,
)
 
# --------------------------------------------------------------------------
# Workspace Section
# --------------------------------------------------------------------------
col_upload, col_result = st.columns([1, 1], gap="large")
 
# --- LEFT COLUMN: Upload X-Ray ---
with col_upload:
    st.markdown(
        """
        <div class="panel-head">
            <h2 class="section-title">Upload X-Ray</h2>
            <span class="panel-tag">Input</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="field-label">Choose X-Ray Image</p>', unsafe_allow_html=True)
 
    uploaded_file = st.file_uploader(
        "Choose X-Ray Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
 
    if uploaded_file is not None:
        preview_img = Image.open(uploaded_file).convert("RGB")
        st.markdown(
            f"""
            <div class="scan-frame">
                <span class="corner tl"></span><span class="corner tr"></span>
                <span class="corner bl"></span><span class="corner br"></span>
                <img src="{img_to_data_uri(preview_img)}" style="width:100%;" />
            </div>
            <div class="scan-caption">Original X-Ray</div>
            """,
            unsafe_allow_html=True,
        )
 
    st.markdown('<p class="field-label" style="margin-top:24px;">Confidence Threshold</p>', unsafe_allow_html=True)
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
        label_visibility="collapsed"
    )
 
    run_clicked = st.button(
        "Run Detection",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )
 
# --- RIGHT COLUMN: Detection Result ---
with col_result:
    st.markdown(
        """
        <div class="panel-head">
            <h2 class="section-title">Detection Result</h2>
            <span class="panel-tag">Output</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    if uploaded_file is not None and run_clicked:
        with st.spinner("Running inference on X-ray image..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                resp = requests.post(
                    f"{BACKEND_URL}/predict/base64",
                    files=files,
                    params={"conf": conf_threshold},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Cannot reach backend: {exc}")
                data = None
 
        if data:
            annotated_bytes = base64.b64decode(data["annotated_image_base64"])
            annotated_img = Image.open(BytesIO(annotated_bytes))
            st.markdown(
                f"""
                <div class="scan-frame">
                    <span class="corner tl"></span><span class="corner tr"></span>
                    <span class="corner bl"></span><span class="corner br"></span>
                    <img src="{img_to_data_uri(annotated_img)}" style="width:100%;" />
                </div>
                <div class="scan-caption">Detection Output</div>
                """,
                unsafe_allow_html=True,
            )
 
            detections = data.get("detections", [])
            if not detections:
                st.markdown(
                    '<div class="no-detection-box">No fracture detected at the current confidence threshold.</div>',
                    unsafe_allow_html=True,
                )
            else:
                rows = ""
                for det in detections:
                    pct = det["confidence"] * 100
                    rows += f"""
                    <div class="detection-row">
                        <div class="det-name">{det['class_name']}</div>
                        <div class="det-bar-track">
                            <div class="det-bar-fill" style="width:{pct:.1f}%;"></div>
                        </div>
                        <div class="det-pct">{pct:.1f}%</div>
                    </div>
                    """
                st.markdown(
                    f"""
                    <div class="detections-wrap">
                        <div class="detections-header">
                            <span>Finding</span><span>Confidence</span>
                        </div>
                        {rows}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
            <div class="empty-result-box">
                Run detection to display annotated results here.
            </div>
            """,
            unsafe_allow_html=True,
        )
 
# --------------------------------------------------------------------------
# Disclaimer Section
# --------------------------------------------------------------------------
st.markdown(
    """
    <hr class="divider" style="margin-top: 50px;" />
    <div class="disclaimer-section">
        <h3>Notice</h3>
        <p>This AI model is developed for educational and research purposes only.</p>
        <p>It should not be used as a replacement for professional medical diagnosis.</p>
        <p>Always consult a qualified orthopedist or radiologist before making any medical decision.</p>
    </div>
    """,
    unsafe_allow_html=True,
)