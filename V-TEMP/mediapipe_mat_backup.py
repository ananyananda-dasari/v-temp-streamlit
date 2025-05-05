import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.io import savemat

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define 3D model points for head pose estimation
model_points_3d = np.array([
    [0.0, 0.0, 0.0],         # Nose tip (1)
    [0.0, -330.0, -65.0],    # Chin (152)
    [-225.0, 170.0, -135.0], # Left eye (263)
    [225.0, 170.0, -135.0],  # Right eye (33)
    [-150.0, -150.0, -125.0],# Left mouth (287)
    [150.0, -150.0, -125.0]  # Right mouth (57)
], dtype=np.float64)
landmark_indices = [1, 152, 263, 33, 287, 57]

def get_head_pose(landmarks, frame_width, frame_height):
    image_points_2d = []
    for idx in landmark_indices:
        lm = landmarks[idx]
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        image_points_2d.append([x, y])
    image_points_2d = np.array(image_points_2d, dtype="double")
    camera_matrix = np.array([
        [frame_width, 0, frame_width / 2],
        [0, frame_width, frame_height / 2],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(model_points_3d, image_points_2d, camera_matrix, dist_coeffs)
    if not success:
        return [np.nan, np.nan, np.nan]
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return np.radians(euler_angles.flatten())  # pitch, yaw, roll

def process_video_to_mat(video_path, output_mat_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    fit_data = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_idx == 20:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # Extract landmark x,y coordinates
            landmarks_xy = np.array([[lm.x, lm.y] for lm in landmarks])

            # Assemble row as specified
            row = [
                frame_idx,                              # Frame number
                1,                                       # Fixed value
                np.random.rand(49, 2),                   # Garbage 49x2
                landmarks_xy,                            # Landmarks (N x 2)
                get_head_pose(landmarks, w, h),          # Head pose (3,)
                np.random.rand(30, 1)                    # Garbage 30x1
            ]
            fit_data.append(row)

        frame_idx += 1

    cap.release()

    # Save to .mat file
    mat_data = {'fit': np.array(fit_data, dtype=object)}
    savemat(output_mat_path, mat_data)
    print(f"Saved .mat file to {output_mat_path}")

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_fit' + '.mat')
            print(f"Processing {filename}...")
            process_video_to_mat(video_path, output_path)

# === RUN ===
input_folder = "./"
output_folder = "./"
process_folder(input_folder, output_folder)
