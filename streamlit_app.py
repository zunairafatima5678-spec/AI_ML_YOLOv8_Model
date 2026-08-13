import streamlit as st
import cv2
from ultralytics import YOLO
import os
import tempfile
import gdown

# 1. Model download
os.makedirs("models", exist_ok=True)
model_path = "models/best.pt"
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/uc?id=17jn7a7e_vQtEr4pR4itXZchXQenL888U", model_path, quiet=False)

model = YOLO(model_path)

st.set_page_config(page_title="YOLOv8 Vehicle Detection", page_icon="🚗")
st.title("🚗 YOLOv8 Vehicle Detection")

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    st.video(uploaded_file, format='video/mp4')
    
    # 2. یہ والا طریقہ use کریں temp file کے لیے
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    temp_video_path = tfile.name
    
    st.info("Processing... please wait")
    
    # 3. YOLO predict
    results = model.predict(source=temp_video_path, save=True, conf=0.5)
    
    # 4. Output video دکھائیں
    for r in results:
        output_dir = r.save_dir
        output_video_path = os.path.join(output_dir, os.path.basename(temp_video_path))
        
    if os.path.exists(output_video_path):
        st.success("Done!")
        st.video(output_video_path)
    else:
        st.error("Output video not found")


