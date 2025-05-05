import os
import cv2
import numpy as np
import mediapipe as mp
import scipy.io

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Dummy head pose estimator (replace with actual model if needed)
def estimate_head_pose(landmarks):
    return np.random.randn(3)  # dummy [pitch, yaw, roll]

# FitStruct equivalent to MATLAB struct fields
class FitStruct:
    def __init__(self, frame, isTracked, pts_2d, pts_3d, headPose, pdmPars):
        self.frame = frame
        self.isTracked = isTracked
        self.pts_2d = pts_2d
        self.pts_3d = pts_3d
        self.headPose = headPose
        self.pdmPars = pdmPars

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    fit_entries = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == 20:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            pts_3d = np.array([[p.x * w, p.y * h] for p in lm.landmark])
            head_pose = estimate_head_pose(pts_3d)
            pts_2d = np.random.rand(49, 2)  # Dummy
            pdm_pars = np.random.rand(30, 1)  # Dummy

            entry = FitStruct(
                frame=frame_idx,
                isTracked=1,
                pts_2d=pts_2d,
                pts_3d=pts_3d,
                headPose=head_pose,
                pdmPars=pdm_pars
            )
            fit_entries.append(entry)

        frame_idx += 1

    cap.release()
    return fit_entries

def process_all_videos(input_folder, output_file):
    all_fits = []
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing {fname}...")
            path = os.path.join(input_folder, fname)
            all_fits.extend(process_video(path))

    # Create structured numpy array
    dtype = np.dtype([
        ('frame', 'O'),
        ('isTracked', 'O'),
        ('pts_2d', 'O'),
        ('pts_3d', 'O'),
        ('headPose', 'O'),
        ('pdmPars', 'O')
    ])
    mat_struct = np.empty(len(all_fits), dtype=dtype)
    for i, e in enumerate(all_fits):
        mat_struct[i] = (e.frame, e.isTracked, e.pts_2d, e.pts_3d, e.headPose, e.pdmPars)

    scipy.io.savemat(output_file, {'fit': mat_struct})
    print(f"Saved: {output_file}")

# Example usage
# process_all_videos("./", "output_fit.mat")


