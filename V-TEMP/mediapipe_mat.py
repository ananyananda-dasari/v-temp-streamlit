import os
import cv2
import numpy as np
import mediapipe as mp
import scipy.io

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

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

# # Dummy head pose estimator (replace with actual if needed)
# def estimate_head_pose(landmarks):
#     return np.random.randn(3)  # Dummy [pitch, yaw, roll]

# Structure matching MATLAB 'fit' format
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
        if frame_idx == 200:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            pts_3d = np.array([[p.x * w, p.y * h] for p in lm.landmark])
            # head_pose = estimate_head_pose(pts_3d)
            landmarks = results.multi_face_landmarks[0].landmark
            frame_height, frame_width = frame.shape[:2]

            pitch, yaw, roll = get_head_pose(landmarks, frame_width, frame_height)


            pts_2d = np.random.rand(49, 2)  # Dummy
            pdm_pars = np.random.rand(30, 1)  # Dummy

            entry = FitStruct(
                frame=frame_idx,
                isTracked=1,
                pts_2d=pts_2d,
                pts_3d=pts_3d,
                headPose=[pitch, yaw, roll],
                pdmPars=pdm_pars
            )
            fit_entries.append(entry)

        frame_idx += 1

    cap.release()
    return fit_entries

def process_all_videos(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if fname.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing {fname}...")
            video_path = os.path.join(input_folder, fname)
            fits = process_video(video_path)

            dtype = np.dtype([
                ('frame', 'O'),
                ('isTracked', 'O'),
                ('pts_2d', 'O'),
                ('pts_3d', 'O'),
                ('headPose', 'O'),
                ('pdmPars', 'O')
            ])
            mat_struct = np.empty(len(fits), dtype=dtype)
            for i, e in enumerate(fits):
                mat_struct[i] = (e.frame, e.isTracked, e.pts_2d, e.pts_3d, e.headPose, e.pdmPars)

            mat_filename = os.path.splitext(fname)[0] + '_fit.mat'
            mat_path = os.path.join(output_folder, mat_filename)
            scipy.io.savemat(mat_path, {'fit': mat_struct})
            print(f"Saved: {mat_path}")

# Example usage
# Replace 'Videos' with your actual input folder
# input_folder = 'V-TEMP/Input_Videos'
# output_folder = 'V-TEMP/Mat_Files'
# process_all_videos(input_folder, output_folder)
