import os
# import cv2
import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
# import av
import tempfile
import time
from main_temp import *
# from backup1_linux_kfold import get_mp4_files


# Define paths
INPUT_VID_DIR = '/tmp/Input_Videos/'
MAT_PATH = '/tmp/Mat_Files/'
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
    # Save file to a fixed known path in INPUT_VID_DIR
    temp_file_path = os.path.join(INPUT_VID_DIR, uploaded_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.read())
    st.session_state.video_uploaded_path = temp_file_path
    st.success(f"Video uploaded and saved to {temp_file_path}")

if uploaded_file is not None:
    video_path = os.path.join(INPUT_VID_DIR, uploaded_file.name)
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.read())

    if not os.path.exists(video_path):
        st.error(f"[ERROR] Failed to save uploaded video to: {video_path}")
    else:
        st.success(f"Video uploaded and saved to {video_path}")
        st.session_state.video_uploaded_path = video_path
        st.video(video_path)  # preview


# Fever detection
st.header("Step 2: Run Fever Detection")

# files = get_mp4_files(INPUT_VID_DIR)

video_to_process = st.session_state.video_uploaded_path or st.session_state.video_recorded_path
if video_to_process:
    st.info(f"Video selected for processing: {video_to_process}")
    if st.button("Run Fever Detection"):
        with st.spinner("Processing video and analyzing skin temperature..."):
            try:
                # Ensure the video path is available for your backend logic
                # If run_main expects INPUT_VID_DIR, make sure it looks inside it
                state_temp = run_main(INPUT_VID_DIR, MAT_PATH, CSV_DATA)
                st.success(state_temp)
                if state_temp[0] == 0:
                    st.success('Skin temperature within normal range. No sign of fever detected')
                if state_temp[0] == 1:
                    st.success('Skin temperature is elevated. Fever detected.')
                st.success("Fever detection completed successfully.")
            except Exception as e:
                st.error(f"Error during fever detection: {e}")
else:
    st.warning("Please upload or record a video first.")
