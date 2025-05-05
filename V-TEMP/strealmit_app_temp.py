import os
import streamlit as st
import tempfile
from main_temp import *  # Assuming this contains the function for fever detection

# Define paths relative to the Streamlit app
MAT_PATH = '/tmp/Mat_Files/'  # Temporary directory in Streamlit environment
os.makedirs(MAT_PATH, exist_ok=True)

# Session states
if 'video_uploaded_path' not in st.session_state:
    st.session_state.video_uploaded_path = None
if 'mat_file_path' not in st.session_state:
    st.session_state.mat_file_path = None  # Store the generated .mat file path

st.title("🌡️ V-TEMP: Video-based detection of elevated skin temperature")

st.header("Step 1: Provide a video")

# Upload a video
uploaded_file = st.file_uploader("Upload a video (MP4 only):", type=['mp4'])
if uploaded_file is not None:
    # Use temporary directory to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        st.session_state.video_uploaded_path = temp_file.name
        st.success(f"Video uploaded and saved to {temp_file.name}")

# Fever detection
st.header("Step 2: Run Fever Detection")

video_to_process = st.session_state.video_uploaded_path
if video_to_process:
    st.info(f"Video selected for processing: {video_to_process}")
    if st.button("Run Fever Detection"):
        with st.spinner("Processing video and analyzing skin temperature..."):
            try:
                # Process the video and generate .mat files
                state_temp = run_main('/tmp/Input_Videos/', MAT_PATH, '/tmp/Input_Data/test.csv')  # Use temporary paths
                st.success(state_temp)
                
                # Check for .mat file and display its path
                mat_file_path = os.path.join(MAT_PATH, "processed_video_fit.mat")
                if os.path.exists(mat_file_path):
                    st.session_state.mat_file_path = mat_file_path
                    st.success(f"Generated .mat file: {mat_file_path}")

                # Proceed with fever detection logic
                if state_temp[0] == 0:
                    st.success('Skin temperature within normal range. No sign of fever detected.')
                if state_temp[0] == 1:
                    st.success('Skin temperature is elevated. Fever detected.')
                st.success("Fever detection completed successfully.")

            except Exception as e:
                st.error(f"Error during fever detection: {e}")
else:
    st.warning("Please upload or record a video first.")
