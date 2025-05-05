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
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.read())
    st.session_state.video_uploaded_path = video_path
    st.success(f"Video uploaded and saved to {video_path}")

# # Webcam recording
# st.markdown("---")
# st.subheader("Or record a video using your webcam")

# class VideoRecorder(VideoTransformerBase):
#     def __init__(self):
#         self.frames = []
#         self.is_recording = False
#         self.out = None
#         self.recording_start_time = None

#     def start_recording(self):
#         """Starts recording frames"""
#         self.frames = []  # Reset frames when starting a new recording
#         self.is_recording = True
#         self.recording_start_time = time.time()
#         st.session_state.recorded_frames = self.frames  # Store in session state
#         st.info("Recording started...")

#     def stop_recording(self):
#         """Stops the recording and saves the video"""
#         self.is_recording = False
#         if self.frames:
#             h, w, _ = self.frames[0].shape
#             out_path = os.path.join(INPUT_VID_DIR, "recorded_video.mp4")
#             self.out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
#             for frame in self.frames:
#                 self.out.write(frame)
#             self.out.release()
#             st.session_state.video_recorded_path = out_path
#             st.session_state.recorded_frames = []  # Clear frames after saving
#             st.success(f"Video recorded and saved to {out_path}")
#             return out_path
#         else:
#             st.warning("Recording stopped but no frames were captured.")
#             return None

#     def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
#         """Handles video frames and captures them during the recording"""
#         if self.is_recording:
#             img = frame.to_ndarray(format="bgr24")
#             self.frames.append(img)
#         return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

# video_recorder = VideoRecorder()

# # Streamer and video capture
# webrtc_ctx = webrtc_streamer(
#     key="record",
#     mode=WebRtcMode.SENDRECV,
#     video_processor_factory=lambda: video_recorder,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=False,
# )

# # Start Recording Button
# if st.button("Start Recording"):
#     video_recorder.start_recording()

# # Stop Recording Button
# if st.button("Stop and Save Recording"):
#     recorded_video_path = video_recorder.stop_recording()
#     if recorded_video_path:
#         # Preview recorded video
#         st.video(recorded_video_path)
#     else:
#         st.warning("No frames were recorded.")

# st.markdown("---")

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
