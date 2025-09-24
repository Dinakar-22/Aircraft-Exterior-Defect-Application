import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -------------------------------
# Load YOLOv8 model
# -------------------------------
model_path = os.path.join("runs", "train", "aircraft_defect", "weights", "best.pt")
model = YOLO(model_path)

# -------------------------------
# Helpers
# -------------------------------
def get_priority(conf: float) -> str:
    """Return priority level based on confidence score."""
    if conf >= 80:
        return "high"
    elif conf >= 50:
        return "medium"
    else:
        return "low"

def get_highest_risk(defects: list) -> str:
    """Return overall highest risk from all detected defects."""
    if not defects:
        return "safe"
    levels = [get_priority(d["confidence"]) for d in defects]
    if "high" in levels:
        return "high"
    elif "medium" in levels:
        return "medium"
    else:
        return "low"

def defect_pipeline(image):
    """Run YOLO prediction and return annotated image + defects list."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb)
    annotated_img = results[0].plot()

    defects = []
    if hasattr(results[0], "boxes"):
        for box, cls_id, conf in zip(results[0].boxes.xyxy,
                                     results[0].boxes.cls,
                                     results[0].boxes.conf):
            defects.append({
                "class": model.names[int(cls_id)],
                "confidence": round(float(conf) * 100, 2),
                "bbox": [float(x) for x in box]
            })
    return annotated_img, defects

def generate_pdf(defects, highest_risk):
    """Generate a PDF report and return as BytesIO buffer."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(180, height - 50, "Aircraft Defect Report")

    # Summary
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Total Defects: {len(defects)}")
    c.drawString(50, height - 120, f"Highest Risk: {highest_risk.upper()}")

    # Defect details
    y = height - 160
    for i, d in enumerate(defects, 1):
        pr = get_priority(d["confidence"]).capitalize()
        line = f"{i}. Type: {d['class']} | Confidence: {d['confidence']}% | Priority: {pr}"
        c.drawString(50, y, line)
        y -= 20
        if y < 50:  # add new page if content runs out
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# -------------------------------
# Streamlit UI Config
# -------------------------------
st.set_page_config(page_title="Aircraft Defect Detection",
                   layout="wide",
                   page_icon="✈️")

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
/* Full Page Background Animation */
@keyframes gradientAnimation {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
html, body, [class*="stApp"] {
    background: linear-gradient(-45deg, #0d0d0d, #1a1a1a, #121212, #222222);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #f5f5f5;
}

/* Title */
.title { color: #4fc3f7; font-size: 32px; font-weight: bold; text-align: center; margin-bottom: 15px; }
.subtitle { text-align: center; color: #ccc; font-size: 16px; margin-bottom: 40px; }

/* Cards */
.card {
    background: #111111;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
    text-align: center;
    color: #e0e0e0;
}

/* Priority Labels */
.high { background: #b71c1c; color: #fff; font-weight: bold; padding: 5px 12px; border-radius: 8px; }
.medium { background: #e65100; color: #fff; font-weight: bold; padding: 5px 12px; border-radius: 8px; }
.low { background: #1b5e20; color: #fff; font-weight: bold; padding: 5px 12px; border-radius: 8px; }
.safe { background: #2e7d32; color: #fff; font-weight: bold; padding: 5px 12px; border-radius: 8px; }

/* Buttons */
button, .stDownloadButton button {
    background: #0d47a1 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none;
    padding: 10px 20px;
}
button:hover, .stDownloadButton button:hover {
    background: #1565c0 !important;
}

/* Center download buttons inside cards */
.stDownloadButton { display: flex; justify-content: center; }
</style>
""", unsafe_allow_html=True)

# ------------------- Header -------------------
st.markdown('<div class="title">Aircraft Defect Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect cracks, dents, corrosion, and paint damage using AI with precision analysis</div>', unsafe_allow_html=True)

# ------------------- Upload -------------------
uploaded_file = st.file_uploader("📤 Upload an aircraft image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    annotated_img, defects = defect_pipeline(img_array)
    highest_risk = get_highest_risk(defects)

    # ------------------- Summary Cards -------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card"><h4>Total Defects</h4><h2>{len(defects)}</h2></div>', unsafe_allow_html=True)

    with col2:
        if highest_risk == "high":
            st.markdown(f'<div class="card"><h4>Highest Risk</h4><div class="high">High</div></div>', unsafe_allow_html=True)
        elif highest_risk == "medium":
            st.markdown(f'<div class="card"><h4>Highest Risk</h4><div class="medium">Medium</div></div>', unsafe_allow_html=True)
        elif highest_risk == "low":
            st.markdown(f'<div class="card"><h4>Highest Risk</h4><div class="low">Low</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card"><h4>Highest Risk</h4><div class="safe">Safe</div></div>', unsafe_allow_html=True)

    with col3:
        if defects:
            pdf_buf = generate_pdf(defects, highest_risk)
            st.markdown('<div class="card" style="text-align:center;"><h4>Report</h4>', unsafe_allow_html=True)
            st.download_button("📄 Export PDF",
                               data=pdf_buf,
                               file_name="aircraft_defect_report.pdf",
                               mime="application/pdf",
                               key="pdf_button")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;">
                <h4>Report</h4>
                <p style="color:#aaa; margin-top:15px;">No defects to include in report.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------- Layout: Image + Details -------------------
    col_img, col_table = st.columns([2, 1])
    with col_img:
        st.subheader("Detected Defects")
        st.image(annotated_img, use_container_width=True)

    with col_table:
        st.subheader("Defect Details")
        if defects:
            for d in defects:
                priority = get_priority(d["confidence"])
                st.markdown(f"""
                <div class="card" style="margin-bottom:10px; text-align:left">
                    <b>Type:</b> {d['class']}<br>
                    <b>Confidence:</b> {d['confidence']}%<br>
                    <b>Status:</b> <span class="{priority}">{priority.capitalize()} Priority</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No defects detected.")

    # ------------------- Download Annotated -------------------
    buf = io.BytesIO()
    Image.fromarray(annotated_img).save(buf, format="PNG")
    st.markdown('<div style="text-align:center; margin-top:20px;">', unsafe_allow_html=True)
    st.download_button("📥 Download Annotated Image",
                       buf,
                       file_name=f"annotated_{uploaded_file.name}",
                       mime="image/png")
    st.markdown('</div>', unsafe_allow_html=True)
