import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile
import gdown
import subprocess # یہ نئی لائن add کریں

# 1. Model download
os.makedirs("models", exist_ok=True)
model_path = "models/best.pt"
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/uc?id=17jn7a7e_vQtEr4pR4itXZchXQenL888U", model_path, quiet=False)

model = YOLO(model_path)

st.set_page_config(page_title="YOLOv8 Vehicle Detection", page_icon="🚗")
st.title("🚗 YOLOv8 Vehicle Detection")

uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.type
    
    # Image کا case
    if "image" in file_type:
        st.image(file_bytes, caption="Uploaded Image")
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        st.info("Processing...")
        results = model.predict(source=img, conf=0.5)
        
        for r in results:
            annotated_img = r.plot()
            st.image(annotated_img, caption="Detected Image")
    
    # Video کا case - نیا طریقہ
    elif "video" in file_type:
        st.video(file_bytes)
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(file_bytes)
        tfile.close()
        temp_video_path = tfile.name
        
        st.info("Processing... please wait 2-3 minutes")
        progress_bar = st.progress(0)
        
        # 1. پہلے سارے detection والے frames کو فولڈر میں save کریں
        results = model.predict(source=temp_video_path, save=False, conf=0.5, stream=True)
        os.makedirs("frames", exist_ok=True)
        
        frame_count = 0
        for i, r in enumerate(results):
            frame = r.plot()
            frame = cv2.resize(frame, (640, 360))
            cv2.imwrite(f"frames/frame_{i:05d}.jpg", frame)
            frame_count += 1
            progress_bar.progress(min(i/100, 1.0)) # rough progress
        
        # 2. اب ffmpeg سے ان سب images کی video بنا دیں
        output_path = "output.mp4"
        command = [
            'ffmpeg', '-framerate', '10', '-i', 'frames/frame_%05d.jpg', 
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', output_path
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        st.success("Done!")
        st.video(output_path)
        
        # صفائی
        os.remove(temp_video_path)
        for f in os.listdir("frames"): 
            os.remove(os.path.join("frames", f))
        os.rmdir("frames")


