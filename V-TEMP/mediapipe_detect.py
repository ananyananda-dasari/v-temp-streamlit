import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define 3D model points for head pose estimation (based on facial landmarks)
model_points_3d = np.array([
    [0.0, 0.0, 0.0],         # Nose tip (landmark 1)
    [0.0, -330.0, -65.0],    # Chin (landmark 152)
    [-225.0, 170.0, -135.0], # Left eye corner (landmark 263)
    [225.0, 170.0, -135.0],  # Right eye corner (landmark 33)
    [-150.0, -150.0, -125.0],# Left mouth corner (landmark 287)
    [150.0, -150.0, -125.0]  # Right mouth corner (landmark 57)
], dtype=np.float64)

# Indices of landmarks corresponding to the model points
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
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    success, rotation_vector, _ = cv2.solvePnP(model_points_3d, image_points_2d, camera_matrix, dist_coeffs)
    if not success:
        return None, None, None

    # Convert rotation vector to euler angles (pitch, yaw, roll)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch = np.radians(euler_angles[0])
    yaw = np.radians(euler_angles[1])
    roll = np.radians(euler_angles[2])
    
    return pitch, yaw, roll

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]

    results_list = []
    frame_idx = 0

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
            frame_height, frame_width = frame.shape[:2]

            pitch, yaw, roll = get_head_pose(landmarks, frame_width, frame_height)

            row = {
                "frame": frame_idx,
                "pitch": pitch if pitch is not None else np.nan,
                "yaw": yaw if yaw is not None else np.nan,
                "roll": roll if roll is not None else np.nan
            }

            for idx, lm in enumerate(landmarks):
                row[f"x_{idx}"] = lm.x
                row[f"y_{idx}"] = lm.y

            results_list.append(row)

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(results_list)
    output_csv = os.path.join(output_dir, f"{filename}_landmarks_pose.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

# ==== Main Script ====
video_folder = "./"
output_folder = "./"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(video_folder):
    if file.endswith(".mp4") or file.endswith(".avi") or file.endswith(".mov"):
        video_path = os.path.join(video_folder, file)
        process_video(video_path, output_folder)
