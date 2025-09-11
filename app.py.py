# app.py
import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
import uuid
import numpy as np
from collections import Counter
import random

# --- Page Setup ---
st.set_page_config(page_title="Bounce Tennis Shot Tracker", layout="wide")

# --- Header & Instructions ---
st.markdown(
    """
    <div style="text-align: center; padding: 30px;">
        <h1 style="font-size: 3em; color: #2E86C1;">🎾 Bounce Tennis Shot Tracker</h1>
        <p style="font-size: 1.3em; font-style: italic; color: #444;">
            Turn practice into your playground
        </p>
        <hr style="margin: 20px 0;">
        <p style="font-size: 1.1em; color: #333;">
            Follow these 4 simple steps to analyse your tennis practice:
        </p>
        <p style="font-size: 1.05em; color: #555; text-align: left; max-width: 700px; margin: auto;">
            1️⃣ <b>Upload</b> your video (<code>.mp4</code>, <code>.mov</code>, <code>.avi</code>)  
            2️⃣ <b>Wait</b> while our AI processes each frame (watch the progress bar)  
            3️⃣ <b>Review</b> your shot breakdown + personalised feedback  
            4️⃣ <b>Download</b> your fully analysed video using the big button below  
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load YOLO model ---
MODEL_PATH = "my_model.pt"
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model at '{MODEL_PATH}': {e}")
    st.stop()

# --- Define class labels (match your training) ---
class_map = {
    0: "Forehand",
    1: "Backhand",
    2: "Serve"
}

# --- Coaching feedback options ---
feedback_tips = [
    "Add more topspin by brushing up the back of the ball on forehands and backhands.",
    "Practice 20 serves in a row focusing only on ball toss consistency.",
    "Hit cross-court forehands for 5 minutes to improve rally reliability.",
    "Work on split-steps before every shot to stay balanced and ready.",
    "Shadow swing 15 backhands slowly to refine technique and footwork.",
    "Do 10 minutes of volley practice, aiming to keep the racquet head stable.",
    "Rally with a partner but only allowed to hit deep to the baseline.",
    "Play '2 cross-court, 1 down the line' to build shot variety.",
    "Practice approach shots followed by finishing volleys at the net.",
    "Focus on recovery steps after each shot to return to the centre quickly."
]

# --- Upload video ---
uploaded_file = st.file_uploader("🎥 Upload a tennis video", type=["mp4", "mov", "avi"])
if uploaded_file is None:
    st.info("Please upload a video to begin.")
    st.stop()

# Save uploaded file
temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
temp_in.write(uploaded_file.read())
temp_in.close()
video_path = temp_in.name

# Output path
out_filename = f"processed_{uuid.uuid4().hex}.mp4"
out_path = os.path.join(tempfile.gettempdir(), out_filename)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error("❌ Could not open video. Please try a standard MP4 (H.264).")
    st.stop()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Track shot counts
shot_counter = Counter()

# Progress
st.subheader("⏳ Processing your video")
progress_bar = st.progress(0)
status_text = st.empty()

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, save=False, imgsz=640, verbose=False)
    annotated = results[0].plot()

    # Count detected shots
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in class_map:
            shot_counter[class_map[cls_id]] += 1

    if annotated is None:
        annotated = frame
    if annotated.dtype != np.uint8:
        annotated = (255 * np.clip(annotated, 0, 1)).astype(np.uint8)
    if (annotated.shape[1], annotated.shape[0]) != (width, height):
        annotated = cv2.resize(annotated, (width, height))

    writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    i += 1
    if frame_count > 0:
        progress_bar.progress(min(i / frame_count, 1.0))
        status_text.text(f"Processed frame {i}/{frame_count}")

cap.release()
writer.release()

# --- Results ---
st.subheader("📊 Results")
if shot_counter:
    for shot, count in shot_counter.items():
        st.markdown(f"- **{shot}:** {count}")
else:
    st.markdown("No shots detected. Try uploading a clearer video.")

# Show random coaching tip
st.markdown(
    f"""
    <div style="background-color:#F0F8FF; padding:15px; border-radius:10px; margin-top:20px;">
        <b>💡 Coaching Tip Based On Your Video:</b> {random.choice(feedback_tips)}
    </div>
    """,
    unsafe_allow_html=True
)

# --- Download button ---
if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download Processed Video",
        data=open(out_path, "rb").read(),
        file_name=f"processed_{uploaded_file.name}",
        mime="video/mp4",
        use_container_width=True,
    )