import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import io
import os

# -------------------------------
# Load YOLOv8 model
# -------------------------------
import os

"""model_path = os.path.join("runs", "train", "aircraft_defect", "weights", "best.pt")
model = YOLO(model_path)"""

# -------------------------------
# Pipeline Function
# -------------------------------
def defect_pipeline(image):
    """
    Takes a numpy array image and returns:
    - annotated image with bounding boxes
    - list of detected defects with class, confidence, bbox
    """
    # Convert to RGB (OpenCV uses BGR)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Predict defects
    results = model.predict(img_rgb)
    
    # Annotated image
    annotated_img = results[0].plot()  # returns image with boxes drawn
    
    # Collect detected defects
    defects = []
    if hasattr(results[0], "boxes"):
        for box, cls_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            defects.append({
                "class": model.names[int(cls_id)],
                "confidence": float(conf),
                "bbox": [float(x) for x in box]
            })
    
    return annotated_img, defects

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Aircraft Defect Detection", layout="wide")

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1 {
    color: #0D47A1;
    text-align: center;
}
.upload-section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.image-section {
    display: flex;
    justify-content: space-around;
    margin-top: 20px;
}
.img-container {
    border: 2px solid #0D47A1;
    border-radius: 10px;
    padding: 10px;
    background-color: #ffffff;
}
.table-container {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    margin-top: 20px;
}
button {
    background-color: #0D47A1;
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
}
button:hover {
    background-color: #1565C0;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Title -------------------
st.markdown("<h1>✈️ Aircraft Exterior Defect Detection</h1>", unsafe_allow_html=True)

# ------------------- Upload Section -------------------
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an aircraft image", type=["jpg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Run your YOLO defect pipeline
    annotated_img, defects = defect_pipeline(img_array)
    
    # ------------------- Image Display -------------------
    st.markdown('<div class="image-section">', unsafe_allow_html=True)
    st.markdown('<div class="img-container">Original Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('<div class="img-container">Detected Defects</div>', unsafe_allow_html=True)
    st.image(annotated_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ------------------- Defect Table -------------------
    if defects:
        st.markdown('<div class="table-container">', unsafe_allow_html=True)
        st.subheader("Defects Detected")
        df = pd.DataFrame(defects)
        st.dataframe(df)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("✅ No defects detected.")
    
    # ------------------- Save & Download -------------------
    save_folder = "output"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, uploaded_file.name)
    cv2.imwrite(save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    
    buf = io.BytesIO()
    Image.fromarray(annotated_img).save(buf, format="PNG")
    st.download_button("Download Annotated Image", buf, file_name=f"annotated_{uploaded_file.name}", mime="image/png")