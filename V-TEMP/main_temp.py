from mediapipe_mat import *
from backup1_linux_kfold import *


input_path = './Input_Videos/'
mat_path = './Mat_Files/'
csv_data = './Input_Data/test.csv'

def run_main(input_path, mat_path, csv_data):
    for root,dirs,files in os.walk(input_path):
        videonames=[ _ for _ in files if _.endswith('.mp4') ]
        print(f'Found {len(videonames)} videos in the input folder \n')
    for i in range(0, len(videonames)):
        vidpath = os.path.join(input_path, videonames[i])
        print(vidpath)
        print('Identifying Face and Obtaining Facial Landmarks')
        process_all_videos(input_path, mat_path)
    
    print('Analyzing subject skin temperature')
    status = main(experiment_status = False, experiment='Landmark')

    return status
    # if status[0] == 0:
    #     print('Skin temperature within normal range. No sign of fever detected')
    # if status[0] == 1:
    #     print('Skin temperature is elevated. Fever detected.')
    

# run_main(input_path, mat_path, csv_data)

