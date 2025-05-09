from mediapipe_mat import *
from backup1_linux_kfold import *
import os


input_path = '/tmp/Input_Videos/'
mat_path = '/tmp/Mat_Files/'
csv_data = 'V-TEMP/Input_Data/test.csv'

def run_main(input_path, mat_path, csv_data):
    # Collect all mp4 video files
    videonames = []
    for root, dirs, files in os.walk(input_path):
        videonames.extend([f for f in files if f.endswith('.mp4')])

    print(f'Found {len(videonames)} videos in the input folder')

    if not videonames:
        raise ValueError("No MP4 video files found in the input folder.")

    print('Identifying Face and Obtaining Facial Landmarks')
    process_all_videos(input_path, mat_path)  # <-- call once here

    saved_files = [f for f in os.listdir(mat_path) if f.endswith('.mat')]
    print(f'Saved {len(saved_files)} .mat files to {mat_path}')

    for videoname in videonames:
        vidpath = os.path.join(input_path, videoname)
        print(f'Processed: {vidpath}')
        vid_name_w_ext = os.path.splitext(videoname)[0]

        print('Analyzing subject skin temperature')
        threshold = 300.5
        status = temp_test_main(vid_name_w_ext, total_frames=100, landmark_num=9, frame_limit='Limit', component='B', pixel_padding=5, colorspace='RGB', colospace_component='B')
        final_temps = np.asarray(status)
        # print(np.shape(final_temps))
        pred_scores = final_temps

        y_pred = [1 if score >= threshold else 0 for score in pred_scores]
    return y_pred

# def run_main(input_path, mat_path, csv_data):
#     for root,dirs,files in os.walk(input_path):
#         videonames=[ _ for _ in files if _.endswith('.mp4') ]
#         print(f'Found {len(videonames)} videos in the input folder \n')
#     for i in range(0, len(videonames)):
#         vidpath = os.path.join(input_path, videonames[i])
#         print(vidpath)
#         print('Identifying Face and Obtaining Facial Landmarks')
#         process_all_videos(input_path, mat_path)
    
#     print('Analyzing subject skin temperature')
#     status = main(experiment_status = False, experiment='Landmark')

#     return status
    # if status[0] == 0:
    #     print('Skin temperature within normal range. No sign of fever detected')
    # if status[0] == 1:
    #     print('Skin temperature is elevated. Fever detected.')
    

# run_main(input_path, mat_path, csv_data)

