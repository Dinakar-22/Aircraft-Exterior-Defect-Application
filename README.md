# ✈️ Aircraft Exterior Defect Detection

An AI-powered web application that identifies defects on aircraft exteriors using computer vision and deep learning. Built with Python, Streamlit, and YOLOv8, this tool streamlines visual inspections and enhances safety protocols in aviation maintenance.

## 🚀 Features

- 🔍 Real-time defect detection using YOLOv8
- 📷 Upload and analyze aircraft images
- 📊 Streamlit-based interactive UI
- 🧠 Custom-trained model for exterior defect classification
- 🧪 Testing pipeline for validation and debugging

## 🧰 Tech Stack

- Python 3
- YOLOv8 for object detection
- Streamlit for web interface
- LangChain (optional, for future enhancements)
- OpenCV, NumPy, Pandas

## 📁 Project Structure

├── app.py                   # Streamlit app entry point  
├── train.py                 # Model training script  
├── test.py                  # Model testing script  
├── yolov8n.pt               # Pre-trained YOLOv8 model  
├── requirements.txt         # Python dependencies  
├── README.md                # Project documentation  
├── LICENSE                  # MIT License  
├── runs/                    # YOLO training outputs  
└── train/aircraft_defect/   # Training dataset  

## demo 

<a href="https://aircraft-exterior-defect-detection.streamlit.app/" style="display:inline-block;padding:8px 16px;background-color:#007bff;color:#fff;border-radius:4px;text-decoration:none;">View 

## 🖥️ Getting Started

### 1. Clone the Repository

git clone https://github.com/Dinakar-22/Aircraft-Exterior-Defect-Application.git  
cd Aircraft-Exterior-Defect-Application

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Launch the App

streamlit run app.py

### 4. Train the Model (Optional)

python train.py

## 📸 Sample Workflow

1. Upload an aircraft image via the Streamlit interface.  
2. The YOLOv8 model processes the image and highlights detected defects.  
3. View results and confidence scores instantly.

## 📜 License

This project is licensed under the MIT License <a href="https://github.com/Dinakar-22/Aircraft-Exterior-Defect-Application/tree/main?tab=MIT-1-ov-file" style="display:inline-block;padding:8px 16px;background-color:#007bff;color:#fff;border-radius:4px;text-decoration:none;">View MIT License</a>


## 👨‍💻 Author

**Dinakar-22**  
Passionate about AI, computer vision, and building intuitive user interfaces for real-world impact.

