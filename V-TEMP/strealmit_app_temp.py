import os
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import tempfile
import time
from main_temp import *

# Define paths
INPUT_VID_DIR = 'V-TEMP/Input_Videos/'
MAT_PATH = 'V-TEMP/Mat_Files/'
CSV_DATA = 'V-TEMP/Input_Data/test.csv'

os.makedirs(INPUT_VID_DIR, exist_ok=True)
os.makedirs(MAT_PATH, exist_ok=True)

# Session states
if 'video_uploaded_path' not in st.session_state:
    st.session_state.video_uploaded_path = None
if 'video_recorded_path' not in st.session_state:
    st.session_state.video_recorded_path = None
if 'recorded_frames' not in st.session_state:
    st.session_state.recorded_frames = []

st.title("🌡️ V-TEMP: Video-based detection of elevated skin temperature")

st.header("Step 1: Provide a video")

# Upload a video
uploaded_file = st.file_uploader("Upload a video (MP4 only):", type=['mp4'])
if uploaded_file is not None:
    video_path = os.path.join(INPUT_VID_DIR, uploaded_file.name)
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    # Read and write only first 200 frames
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_count = 0
    while cap.isOpened() and frame_count < 200:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    os.remove(temp_video_path)

    st.session_state.video_uploaded_path = video_path
    st.success(f"Video uploaded and trimmed to 200 frames: {video_path}")

# Fever detection
st.header("Step 2: Run Fever Detection")

video_to_process = st.session_state.video_uploaded_path or st.session_state.video_recorded_path
if video_to_process:
    st.info(f"Video selected for processing: {video_to_process}")
    if st.button("Run Fever Detection"):
        with st.spinner("Processing video and analyzing skin temperature..."):
            try:
                state_temp = run_main(INPUT_VID_DIR, MAT_PATH, CSV_DATA)
                if state_temp[0] == 0:
                    st.success('Skin temperature within normal range. No sign of fever detected')
                if state_temp[0] == 1:
                    st.success('Skin temperature is elevated. Fever detected.')
                st.success("Fever detection completed successfully.")
            except Exception as e:
                st.error(f"Error during fever detection: {e}")
else:
    st.warning("Please upload or record a video first.")
