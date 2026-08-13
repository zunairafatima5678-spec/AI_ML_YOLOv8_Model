import os
import gdown
import streamlit as st
from ultralytics import YOLO

os.makedirs("models", exist_ok=True)  # یہ لائن فولڈر بنا دے گی

model_path = "models/best.pt"
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/file/d/17jn7a7e_vQtEr4pR4itXZchXQenL888U/view?usp=sharing", model_path, quiet=False)

model = YOLO(model_path)

st.set_page_config(page_title="YOLOv8 Vehicle Detection", page_icon="🚗")

st.title("🚗 YOLOv8 Vehicle Detection")
st.write("Upload a video to detect vehicles using a custom-trained YOLOv8n model.")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'models/best.pt'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please ensure it's in the 'models' folder.")
        st.stop()
    return YOLO(model_path)

model = load_model()

uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    st.video(uploaded_file, format='video/mp4', start_time=0)

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(uploaded_file.read())
        temp_video_path = temp_video_file.name

    st.write("Performing object detection...")

    # Perform inference
    # Ultralytics will save results in 'runs/detect/predict-N'
    results = model.predict(source=temp_video_path, save=True, conf=0.25, device='cpu', stream=True)

    output_video_path = None
    output_dir = None

    for r in results:
        # Check for save_dir attribute in the first result object
        if hasattr(r, 'save_dir'):
            output_dir = r.save_dir
            break

    if output_dir:
        # Find the processed video file in the output directory
        # Ultralytics often names the output video with the original source filename
        original_video_basename = os.path.basename(uploaded_file.name)
        potential_output_path = os.path.join(output_dir, original_video_basename)

        if os.path.exists(potential_output_path):
            output_video_path = potential_output_path
        else:
            # Fallback: search for any .mp4 or .avi in the output directory
            video_files = [f for f in os.listdir(output_dir) if f.endswith(('.mp4', '.avi'))]
            if video_files:
                output_video_path = os.path.join(output_dir, video_files[0])

    if output_video_path and os.path.exists(output_video_path):
        st.success("Detection complete!")
        st.video(output_video_path)

        # Optional: Add a download button for the processed video
        with open(output_video_path, "rb") as file:
            btn = st.download_button(
                label="Download Processed Video",
                data=file,
                file_name=f"detected_{uploaded_file.name}",
                mime="video/mp4"
            )
    else:
        st.error("Could not find the processed video file.")

    # Clean up temporary files
    os.remove(temp_video_path)
    if output_dir and os.path.exists(output_dir):
        # Use shutil.rmtree to remove the entire directory and its contents
        shutil.rmtree(output_dir, ignore_errors=True)
        st.info(f"Cleaned up inference directory: {output_dir}")

