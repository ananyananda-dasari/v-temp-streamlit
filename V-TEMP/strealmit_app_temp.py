import os
import streamlit as st
import tempfile
import time
from main_temp import *  # Assuming this contains the function for fever detection

# Define paths
INPUT_VID_DIR = 'V-TEMP/Input_Videos/'
MAT_PATH = 'V-TEMP/Mat_Files/'
CSV_DATA = 'V-TEMP/Input_Data/test.csv'

# Create directories if they don't exist
os.makedirs(INPUT_VID_DIR, exist_ok=True)
os.makedirs(MAT_PATH, exist_ok=True)

# Session states
if 'video_uploaded_path' not in st.session_state:
    st.session_state.video_uploaded_path = None
if 'video_recorded_path' not in st.session_state:
    st.session_state.video_recorded_path = None
if 'recorded_frames' not in st.session_state:
    st.session_state.recorded_frames = []
if 'mat_file_path' not in st.session_state:
    st.session_state.mat_file_path = None  # Store the generated .mat file path

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

video_to_process = st.session_state.video_uploaded_path or st.session_state.video_recorded_path
if video_to_process:
    st.info(f"Video selected for processing: {video_to_process}")
    if st.button("Run Fever Detection"):
        with st.spinner("Processing video and analyzing skin temperature..."):
            try:
                # Process the video and generate .mat files
                state_temp = run_main(INPUT_VID_DIR, MAT_PATH, CSV_DATA)
                st.success(state_temp)
                
                # Assuming run_main generates and saves the .mat file in MAT_PATH
                mat_filename = f"{os.path.splitext(os.path.basename(video_to_process))[0]}_fit.mat"
                mat_file_path = os.path.join(MAT_PATH, mat_filename)

                # Check if the .mat file is generated and stored
                if os.path.exists(mat_file_path):
                    st.session_state.mat_file_path = mat_file_path
                    st.success(f"Generated .mat file: {mat_file_path}")

                    # Proceed with fever detection logic
                    if state_temp[0] == 0:
                        st.success('Skin temperature within normal range. No sign of fever detected.')
                    if state_temp[0] == 1:
                        st.success('Skin temperature is elevated. Fever detected.')
                    st.success("Fever detection completed successfully.")
                else:
                    st.error(f"Error: {mat_file_path} not found after processing video.")

            except Exception as e:
                st.error(f"Error during fever detection: {e}")
else:
    st.warning("Please upload or record a video first.")
