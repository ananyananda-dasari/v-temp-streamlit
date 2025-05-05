import numpy as np 
import cv2
from numpy.core.fromnumeric import shape
import scipy.io
import pandas as pd
import os 
import os.path
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt
from tqdm import tqdm 
import os.path
from os import path
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# def tvalue(imagefile, limit):
#     if limit == 'maxt':
#         top, bottom, left, right = 0, 20, 662, 687 
#     if limit == 'mint':
#         top, bottom, left, right = 460, 480, 662, 692
#     tvalue = []
#     imagefile =  imagefile[top:bottom+5, left:right, :]
#     imagefile = cv2.resize(imagefile,(0,0),fx=3,fy=3)
#     text = pytesseract.image_to_string(imagefile)
#     if limit == 'maxt':    
#         try:
#             tvalue.append((float(text)))
#         except:
#             tvalue.append((37.0))
#     if limit == 'mint':    
#         try:
#             tvalue.append((float(text)))
#         except:
#             tvalue.append((25.0))        
#     return tvalue

# def colorbar(imgcolorbar):
#     topc, bottomc, leftc, rightc = 0, 480, 630, 690 
#     clrbar = []
#     x = 25
#     y = 80
#     imgcolorbar =  imgcolorbar[topc:bottomc, leftc:rightc, :]
#     for i in range(6,472):
#         clrbar.append((imgcolorbar[i, x, :]))
#     return clrbar

# def mindist(pixel_input, compcb):
#     d = [100]*len(compcb)
#     for i in range(0, len(compcb)):
#         dist = (pixel_input[0]-compcb[i][0])**2 + (pixel_input[1]-compcb[i][1])**2 + (pixel_input[2]-compcb[i][2])**2
#         d[i] = np.sqrt(dist)
#     return np.argmin(d).astype(np.uint8)

# def average_thermal(input_frame, x, y, numel):
#     a = []
#     b = []
#     c = []
#     for x_ct in range(x-numel, x+numel):
#         for y_ct in range(y-numel, y+numel):
#             a.append((int(input_frame[x_ct, y_ct, 0])))
#             b.append((int(input_frame[x_ct, y_ct, 1])))
#             c.append((int(input_frame[x_ct, y_ct, 2])))
#     return [np.mean(a), np.mean(b), np.mean(c)]

def average_refl(input_frame, x, y, numel):
    a = []
    for x_c in range(x-numel, x+numel):
        for y_c in range(y-numel, y+numel):
            a.append((int(input_frame[x_c, y_c])))
            
    return np.mean(a)

def face_landmarks(data, current_frame, num, pattern, pattern_arr):
    x_coord_arr = []
    y_coord_arr = []
    if pattern == None:            
        for i in range(0, num):
            x_coord_arr.append(int(data["fit"][int(current_frame)][3][i][0]))
            y_coord_arr.append(int(data["fit"][int(current_frame)][3][i][1]))
    elif pattern == 'array':
        for i in pattern_arr:
            x_coord_arr.append(int(data["fit"][int(current_frame)][3][i][0]))
            y_coord_arr.append(int(data["fit"][int(current_frame)][3][i][1]))

    return x_coord_arr, y_coord_arr


def check_bounds(coordinate_arr_x, coordinate_arr_y, frame_width, frame_height, padding):

    for i in range (0, len(coordinate_arr_x)):
        if coordinate_arr_x[i] >= (frame_height - padding):
            coordinate_arr_x[i] = frame_height - padding
            continue
        if coordinate_arr_x[i] < padding: 
            coordinate_arr_x[i] = padding
            continue
        if coordinate_arr_x[i] < (frame_height - padding) and coordinate_arr_x[i] > padding:
            coordinate_arr_x[i] = coordinate_arr_x[i]
            continue

    for i in range (0, len(coordinate_arr_y)):
        if coordinate_arr_y[i] >= (frame_width - padding): 
            coordinate_arr_y[i] = frame_width - padding
            continue
        if coordinate_arr_y[i] < padding:
            coordinate_arr_y[i] = padding
            continue
        if coordinate_arr_y[i] < (frame_width - padding) and coordinate_arr_y[i] > padding:
            coordinate_arr_y[i] = coordinate_arr_y[i]
            continue
    return coordinate_arr_x, coordinate_arr_y


def refl_calculator(x_arr, y_arr, frame, padding):
    average_refl_arr = []
    for i in range(0, len(x_arr)):
        average_refl_arr.append(average_refl(frame, y_arr[i], x_arr[i], padding))
    return np.mean(average_refl_arr)


def temp_test_main(filename, total_frames, landmark_num, frame_limit, component, pixel_padding, colorspace, colospace_component):
    # directory_mat = r'/media/sakthi/4 TB Hard Drive/Sakthi/Gates/Temperature/Edited_Videos_Rotated_Data/'       ########### Linux
    # vid_dir = r'/media/sakthi/4 TB Hard Drive/Sakthi/Gates/Temperature/Edited_Videos_Rotated/'

    directory_mat = r'V-TEMP/Mat_Files/'       ########### Linux
    vid_dir = r'V-TEMP/Input_Videos/'


    matfilename = directory_mat + filename + '_fit.mat'                 ########### Linux
    if path.isfile(matfilename) == True:
        mat = scipy.io.loadmat(matfilename)
        mat = {k:v for k, v in mat.items() if k[0] != '_'}
        data_2d = pd.DataFrame({k: v[0] for k, v in mat.items()})
        videofilename = vid_dir + filename + '.mp4'
        cap = cv2.VideoCapture(videofilename) 
        fps = cap.get(cv2.CAP_PROP_FPS) 
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(frame_count)
        if frame_limit == None:
            frlen = int(frame_count)
            # print('Frames:', frlen)
        elif frame_limit == 'Limit':
            frlen = total_frames
        landmark = landmark_num
        # duration = frame_count/fps.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_name = filename + 'output_video.avi'
        out = cv2.VideoWriter(output_video_name,fourcc, 25.0, (480, 726))
        reflectance = []
        frameavrefl = []
        euler = []
        ii = 0
        while (cap.isOpened() and ii<frlen-2): 
            ii = ii + 1
        
            ret, frame = cap.read() 
            if ret==True:

                currframe = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                if np.shape(data_2d["fit"][int(currframe)][3])[0] != 0:     ## change [3] to [2] for 512 to 49
                    x_corr, y_corr = np.asarray(face_landmarks(data_2d, currframe, 49, pattern = 'array', pattern_arr=[landmark]))
                    x_corr, y_corr = check_bounds(x_corr, y_corr, np.shape(frame)[0], np.shape(frame)[1], pixel_padding)
                    # frameavrefl.append(refl_calculator(x_corr, y_corr, frame, padding=pixel_padding))
                    # frameavrefl.append((np.mean([average_refl(frame, x1, y1, 3), average_refl(frame, x2, y2, 3), average_refl(frame, x3, y3, 3), average_refl(frame, x4, y4, 3), average_refl(frame, x5, y5, 3)]))) # RGB Reflectance
                    # frameavrefl.append((average_refl(frame, x_all, y_all, 3)))
                    frame_value = []
                    if 'RGB' in colorspace:
                        frame1 = frame
                        if 'B' in colospace_component:
                            frame_value.append(frame1[:, :, 0])
                        if 'G' in colospace_component:
                            frame_value.append(frame1[:, :, 1])
                        if 'R' in colospace_component:
                            frame_value.append(frame1[:, :, 2])
                        
                    if 'LAB' in colorspace:
                        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                        # print(np.shape(frame2))
                        if 'L' in colospace_component:
                            frame_value.append(frame2[:, :, 0])
                        if 'A' in colospace_component:
                            frame_value.append(frame2[:, :, 1])
                        if 'B_l' in colospace_component:
                            frame_value.append(frame2[:, :, 2])

                    if 'HSV' in colorspace:
                        frame3 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        # print(np.shape(frame3))
                        if 'H' in colospace_component:
                            frame_value.append(frame3[:, :, 0])
                        if 'S' in colospace_component:
                            frame_value.append(frame3[:, :, 1])
                        if 'V' in colospace_component:
                            frame_value.append(frame3[:, :, 2])

                    if 'YCrCb' in colorspace:
                        frame4 = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                        # print(np.shape(frame4))
                        if 'Y' in colospace_component:
                            frame_value.append(frame4[:, :, 0])
                        if 'Cr' in colospace_component:
                            frame_value.append(frame4[:, :, 1])
                        if 'Cb' in colospace_component:
                            frame_value.append(frame4[:, :, 2])

                    # print(np.shape(frame_value))
                    frame_final = np.sum(frame_value, axis = 0)
                    
                    
                    reflectance.append(refl_calculator(x_corr, y_corr, frame_final, padding=pixel_padding))
                    for i in range(0, len(x_corr)):
                        frame = cv2.circle(frame, (x_corr[i], y_corr[i]),5,(0,0,255), thickness=5)

                    frame=cv2.resize(frame,(480, 726))
                    out.write(frame)

                    # Display the resulting frame 
                    # cv2.imshow('Frame', frame) 

                    euler.append((data_2d["fit"][int(currframe)][4][1]*180.0/np.pi)) 

                    # euler.append((data_2d["fit"][int(currframe)][4][0][1]*180.0/np.pi))


                    # eulerframe = data_2d["fit"][int(currframe)][4][1]*180.0/np.pi


                if np.shape(data_2d["fit"][int(currframe)][3])[0] == 0:
                    # reflectance.append((reflectance[int(currframe-1)]))
                    if len(reflectance) != 0:
                        reflectance.append((np.mean(reflectance)))
                        # euler.append((eulerframe))
                        euler.append((np.mean(euler)))
                    if len(reflectance) == 0:
                        reflectance.append((np.array(135)))
                        euler.append((np.array(1)))
                    # euler.append((euler[int(currframe)-1]))

            else:
                break
                    
        cap.release() 
        out.release()
        cv2.destroyAllWindows() 
        # print('Reflectance Statistics (Mean, Max, Min):', np.mean(reflectance), max(reflectance), min(reflectance))                   ############ Print Original
        # print('Reflectance Frame Average:', np.mean(frameavrefl))                                                                     ############ Print Original
        reflectance_name = filename + 'Reflectance.csv'
        np.savetxt(reflectance_name, reflectance, delimiter=',')
        # print('Yaw angles:', euler)
        euler_name = filename + 'Euler.csv'
        # euler = np.concatenate(euler, axis=0)
        # np.savetxt(euler_name, euler, delimiter=',')
        eulerdiff = []
        for i in range(0, len(euler)):
            eulerdiff.append((abs(euler[i] - euler[0])))
        
        # for i in range(0, len(eulerdiff)):
        #     if int(eulerdiff[i]) >= 30:
                # print('Found:', filename)
                # movementfile = dirname_move + filename + '.txt' 
                # np.savetxt(movementfile, [filename], fmt = '%s')
        # print(np.shape(eulerdiff))
        # print(eulerdiff)
        # euler_diff_name = filename + 'Euler_diff.csv'
        # np.savetxt(euler_diff_name, eulerdiff, delimiter=',')
        # print(len(reflectance))


        ### PLOTTING ###

        import matplotlib.pyplot as plt
        x = np.linspace(0, frlen, frlen)
        tau = 0.95
        ##############################
        yp2 = []
        for i in range(0, len(reflectance)):
            if eulerdiff[i]<=18:
                yp2.append(((0.99*np.add(tau*(np.add(np.asarray(0.1*reflectance[i]), -1)), 1))/(0.98*np.add(tau*(np.add(np.asarray(0.1*reflectance[0]), -1)), 1))))
            if 18<eulerdiff[i]<=28:
                yp2.append(((0.99*np.add(tau*(np.add(np.asarray(0.2*np.asarray(reflectance[i])), -1)), 1))/(0.98*np.add(tau*(np.add(np.asarray(np.asarray(0.1*reflectance[0])), -1)), 1))))
            if 28<eulerdiff[i]<=38:
                yp2.append(((0.99*np.add(tau*(np.add(np.asarray(0.3*np.asarray(reflectance[i])), -1)), 1))/(0.98*np.add(tau*(np.add(np.asarray(np.asarray(0.1*reflectance[0])), -1)), 1))))
            if 38<eulerdiff[i]:
                yp2.append(((0.99*np.add(tau*(np.add(np.asarray(0.5*np.asarray(reflectance[i])), -1)), 1))/(0.98*np.add(tau*(np.add(np.asarray(np.asarray(0.1*reflectance[0])), -1)), 1))))
        # print("The value:", yp2)
        # yp2 = np.add(0.99*np.asarray(reflectance[0:frlen]), 1)/np.add(0.98*np.asarray(reflectance[0]), 1)
        # print('Reflectance Ratio Statistics (Mean, Max, Min):', np.mean(yp2), np.median(yp2), min(yp2))                       ############ Print Original
        # yp2 = (yp2 - np.mean(yp2))/np.std(yp2)
        # yp2mm = (yp2 - min(yp2))/(max(yp2) - min(yp2))
        # print('Reflectance Ratio MinMax Statistics (Mean, Max, Min):', np.mean(yp2mm), max(yp2mm), min(yp2mm))                ############ Print Original

        #######################################################################################################

        #plt.plot(x, yp1, label='Temperature')
        # plt.plot(x, yp2, label='Reflectance Ratio')
        # #plt.plot(x, k, label='k')
        # ax.set_xlabel('Frame Number')
        # ax.set_ylabel('Reflectance Ratio')
        # ax.legend()
        # fig_1 = filename + 'fig_1.jpg'
        # plt.savefig(fig_1, dpi=300)
        # # plt.show()

        # #plt.plot(x, yp1mm, label='Temperaturemm')
        # plt.plot(x, yp2mm, label='Reflectance Normalized')
        # ax.set_xlabel('Frame Number')
        # ax.set_ylabel('Reflectance Ratio Normalized')
        # #plt.plot(x, kmm, label='kmm')
        # ax.legend()
        # fig_2 = filename + 'fig_2.jpg'
        # plt.savefig(fig_2, dpi=300)
        # plt.show() ### Uncomment to see plots
        # plt.close('all')

        ### Determining Temperature

        # ratio = pd.read_csv(directory_mat + 'Interpolated_Ratio_new.csv', header=None)          ########### Linux
        # ratio_d = ratio.to_numpy()

        ratio = pd.read_csv(directory_mat + 'test.csv', header=None)          ########### Linux
        ratio_d = ratio.to_numpy()

        minratiodist = []
        for i in range(0, len(reflectance)):
            ratio_dist = []
            for j in range(1, np.shape(ratio_d)[0]):
                ratio_dist.append((abs(ratio_d[j, int(eulerdiff[i])+1] - yp2[i]**(0.25))))
            minratiodist.append((ratio_d[np.argmin(ratio_dist)+1, 0]))
        # print(np.mean(minratiodist))        ############ Print Original
        

        # ratio_dist_max = []
        # minratiodist_max = []
        # for j in range(1, np.shape(ratio_d)[0]):
        #     ratio_dist_max.append((abs(ratio_d[j, int(eulerdiff[np.argmax(yp2)])+1] - np.median(yp2)**(0.25))))
        # minratiodist_max = ratio_d[np.argmin(ratio_dist_max)+1, 0]
        # print('Max Temp. =', minratiodist_max)

        # ratio_dist_ang = []
        # # ratio_dist_ang2 = []
        # # ratio_dist_ang3 = []
        # minratiodist_ang = []
        # # minratiodist_ang2 = []
        # # minratiodist_ang3 = []
        # for j in range(1, np.shape(ratio_d)[0]):
        #     ratio_dist_ang.append((abs(ratio_d[j, int(max(eulerdiff))+1] - yp2[np.argmax(eulerdiff)]**(0.25))))
        #     # ratio_dist_ang2.append((abs(ratio_d[j, int(max(eulerdiff))+2] - yp2[np.argmax(eulerdiff)+1]**(0.25))))
        #     # ratio_dist_ang3.append((abs(ratio_d[j, int(max(eulerdiff))-1] - yp2[np.argmax(eulerdiff)+2]**(0.25))))
        # minratiodist_ang = ratio_d[np.argmin(ratio_dist_ang)+1, 0]
        # # minratiodist_ang2 = ratio_d[np.argmin(ratio_dist_ang2)+1, 0]
        # # minratiodist_ang3 = ratio_d[np.argmin(ratio_dist_ang3)+1, 0]
        # print('Max Temp.2 =', np.mean([minratiodist_ang]))

        # return [np.mean(minratiodist), minratiodist_max, minratiodist_ang]
        return [np.mean(minratiodist)]



# dir = r'/media/newhd/Sakthi/Gates/Temperature/PR_Testing'
# dir = r'/media/sakthi/4 TB Hard Drive/Sakthi/Gates/Temperature/Edited_Videos_Rotated/'       ################### Linux

dir = r'V-TEMP/Input_Videos/'       ################### Linux

# dir = r'D:\CMU\BMGF\Temp_Test\3d-facial-landmark-detection-and-tracking-master\3d-facial-landmark-detection-and-tracking-master\NoFeverTest'

allfiles = sorted(os.listdir(dir))
files = [fname for fname in allfiles if fname.endswith('.mp4')]
print(files)
print(len(files))
filenames_list = []


def randomizer(file_names, sample_size):
    files_random = []
    y_true_random = []
    for rand_sample in range(0, sample_size):
        random_index = random.randrange(len(file_names))
        # y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        y_true = pd.read_csv('Ground_Truth.csv', header=None)
        y_true = np.squeeze(np.asarray(y_true))
        # y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        y_true_random.append(y_true[random_index])
        files_random.append(file_names[random_index])
    return files_random, y_true_random


def frame_exp(file_list, y_true_list, frame_size_range, landmark_num, threshold, padding_size, colorspace_comp):
    frame_f1 = []
    frame_range = frame_size_range
    f1_sc = []
    for frame_size in tqdm(frame_range):
        # f1_sc = []
        final_temps = []

        for filename in tqdm(file_list):
            filename_split = os.path.splitext(filename)[0]
            # print(filename_split)        ############ Print Original
            # if filename_split == 'Video21':
            final_temps.append((temp_test_main(filename_split, total_frames = frame_size, landmark_num = landmark_num, frame_limit='Limit', component = 2, pixel_padding = padding_size, colorspace = ['RGB', 'LAB', 'HSV', 'YCrCb'], colospace_component = [str(colorspace_comp)])))     
            filenames_list.append((filename_split))
        print(final_temps)

        # threshold_range = np.linspace(299.5, 300, 1)

        # for threshold in threshold_range:
        print('Threshold:', threshold)
        ## Metrics
        final_temps = np.asarray(final_temps)
        print(np.shape(final_temps))
        pred_scores = final_temps[:, 0]
        y_pred = [1 if score >= threshold else 0 for score in pred_scores]
        print(np.shape(y_pred))
        print(y_pred)
        average_precision = average_precision_score(y_true_list, y_pred)
        f1 = f1_score(y_true_list, y_pred)
        print(confusion_matrix(y_true_list, y_pred))
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        print('F1 Score:', f1)
        f1_sc.append(f1)
        

    print("F1", np.shape(f1_sc))
    frame_f1.append(f1_sc)
    return frame_f1

def landmark_exp(file_list, y_true_list, frame_size, landmark_range_list, threshold, padding_size, colorspace_comp):
    landmark_f1 = []
    f1_sc = []
    for landmark_number in tqdm(landmark_range_list):
        # f1_sc = []
        final_temps = []

        for filename in tqdm(file_list):
            filename_split = os.path.splitext(filename)[0]
            # print(filename_split)            ############ Print Original
            # if filename_split == 'Video21':
            final_temps.append((temp_test_main(filename_split, total_frames = frame_size, landmark_num = landmark_number, frame_limit='Limit', component = 2, pixel_padding = padding_size, colorspace = ['RGB', 'LAB', 'HSV', 'YCrCb'], colospace_component = [str(colorspace_comp)])))     
            filenames_list.append((filename_split))
        print(final_temps)

        # threshold_range = np.linspace(299.5, 300, 1)
        # threshold = threshold_vals
        # for threshold in threshold_range:
        print('Threshold:', threshold)
        ## Metrics
        final_temps = np.asarray(final_temps)
        print(np.shape(final_temps))
        pred_scores = final_temps[:, 0]

        y_pred = [1 if score >= threshold else 0 for score in pred_scores]
        print(np.shape(y_pred))
        print(y_pred)


        average_precision = average_precision_score(y_true_list, y_pred)
        f1 = f1_score(y_true_list, y_pred)
        print(confusion_matrix(y_true_list, y_pred))
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        print('F1 Score:', f1)
        f1_sc.append(f1)

    print("F1", np.shape(f1_sc))

    landmark_f1.append(f1_sc)
    return landmark_f1

def threshold_exp(file_list, y_true_list, frame_size, landmark_num, threshold_range_list, padding_size, colorspace_comp):
    f1_sc = []
    final_temps = []

    for filename in tqdm(file_list):
        filename_split = os.path.splitext(filename)[0]
        # print(filename_split)          ############ Print Original
        # if filename_split == 'Video21':
        final_temps.append((temp_test_main(filename_split, total_frames = frame_size, landmark_num = landmark_num, frame_limit='Limit', component = 2, pixel_padding = padding_size, colorspace = ['RGB', 'LAB', 'HSV', 'YCrCb'], colospace_component = [str(colorspace_comp)])))     
        filenames_list.append((filename_split))
    print(final_temps)

    threshold_range = threshold_range_list
    for threshold in tqdm(threshold_range):
        print('Threshold:', threshold)
        ## Metrics
        final_temps = np.asarray(final_temps)
        print(np.shape(final_temps))
        pred_scores = final_temps[:, 0]

        y_pred = [1 if score >= threshold else 0 for score in pred_scores]
        print(np.shape(y_pred))
        print(y_pred)


        average_precision = average_precision_score(y_true_list, y_pred)
        ns_fpr, ns_tpr, _ = roc_curve(y_true_list, y_pred)
        roc_plot_name = 'ROC_' + str(threshold) + '.png'
        plt.plot(ns_fpr, ns_tpr)
        plt.savefig(roc_plot_name)
        plt.close()
        # plt.show()
        f1 = f1_score(y_true_list, y_pred)
        print(confusion_matrix(y_true_list, y_pred))
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        print('F1 Score:', f1)
        f1_sc.append(f1)

    print("F1 Threshold", np.shape(f1_sc))
    return f1_sc


def padding_exp(file_list, y_true_list, frame_size, landmark_num, threshold, padding_range_list, colorspace_comp):
    padding_f1 = []
    f1_sc = []
    for padding_number in tqdm(padding_range_list):
        # f1_sc = []
        final_temps = []

        for filename in tqdm(file_list):
            filename_split = os.path.splitext(filename)[0]
            # print(filename_split)         ############ Print Original
            # if filename_split == 'Video21':
            final_temps.append((temp_test_main(filename_split, total_frames = frame_size, landmark_num = landmark_num, frame_limit='Limit', component = 2, pixel_padding = padding_number, colorspace = ['RGB', 'LAB', 'HSV', 'YCrCb'], colospace_component = [str(colorspace_comp)])))     
            filenames_list.append((filename_split))
        print(final_temps)

        # threshold_range = np.linspace(26.25, 27, 1)
        # for threshold in threshold_range:
        print('Threshold:', threshold)
        ## Metrics
        final_temps = np.asarray(final_temps)
        print(np.shape(final_temps))
        pred_scores = final_temps[:, 0]

        y_pred = [1 if score >= threshold else 0 for score in pred_scores]
        print(np.shape(y_pred))
        print(y_pred)


        average_precision = average_precision_score(y_true_list, y_pred)
        f1 = f1_score(y_true_list, y_pred)
        print(confusion_matrix(y_true_list, y_pred))
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        print('F1 Score:', f1)
        f1_sc.append(f1)

    print("F1", np.shape(f1_sc))

    padding_f1.append(f1_sc)
    return padding_f1


def colorspace_exp(file_list, y_true_list, frame_size, landmark_num, threshold, padding_size, colorspace_list):
    colorspace_f1 = []
    f1_sc = []
    for colorspace_number in tqdm(colorspace_list):
        # f1_sc = []
        final_temps = []

        for filename in tqdm(file_list):
            filename_split = os.path.splitext(filename)[0]
            # print(filename_split)         ############ Print Original
            # if filename_split == 'Video21':
            final_temps.append((temp_test_main(filename_split, total_frames = frame_size, landmark_num = landmark_num, frame_limit='Limit', component = 2, pixel_padding = padding_size, colorspace = ['RGB', 'LAB', 'HSV', 'YCrCb'], colospace_component = [str(colorspace_number)])))     
            filenames_list.append((filename_split))
        print(final_temps)

        # threshold_range = np.linspace(26.25, 27, 1)
        # for threshold in threshold_range:
        print('Threshold:', threshold)
        ## Metrics
        final_temps = np.asarray(final_temps)
        print(np.shape(final_temps))
        pred_scores = final_temps[:, 0]

        y_pred = [1 if score >= threshold else 0 for score in pred_scores]
        print(np.shape(y_pred))
        print(y_pred)


        average_precision = average_precision_score(y_true_list, y_pred)
        f1 = f1_score(y_true_list, y_pred)
        print(confusion_matrix(y_true_list, y_pred))
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        print('F1 Score:', f1)
        f1_sc.append(f1)

    print("F1", np.shape(f1_sc))

    colorspace_f1.append(f1_sc)
    return colorspace_f1



def folder_eval(file_list, y_true_list, frames, landmark, threshold, padding, colorspace_comp):
    f1_sc = []
    final_temps = []

    for filename in tqdm(file_list):
        # print(filename)
        filename_split = os.path.splitext(filename)[0]
        # print(filename_split)         ############ Print Original
        # if filename_split == 'Video21':
        final_temps.append((temp_test_main(filename_split, total_frames = int(frames), landmark_num = int(landmark), frame_limit='Limit', component = 2, pixel_padding = int(padding), colorspace = ['RGB', 'LAB', 'HSV', 'YCrCb'], colospace_component = [str(colorspace_comp)])))     
        filenames_list.append((filename_split))
    print(final_temps)

    # threshold_range = np.linspace(299.5, 300, 1)
    # for threshold in threshold_range:
    # print('Threshold:', threshold)
    ## Metrics
    final_temps = np.asarray(final_temps)
    # print(np.shape(final_temps))
    pred_scores = final_temps

    y_pred = [1 if score >= threshold else 0 for score in pred_scores]
    # print(np.shape(y_pred))
    print(y_pred)


    # average_precision = average_precision_score(y_true_list, y_pred)
    # f1 = f1_score(y_true_list, y_pred)
    f1 = 1
    # print(confusion_matrix(y_true_list, y_pred))
    # print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    # print('F1 Score:', f1)
    f1_sc.append(f1)

    # print("F1", np.shape(f1_sc))
    return f1_sc, y_pred


def validation(files_valid, y_valid, frame_range, landmark_range, threshold_range, padding_range, colorspace_range):
    opt_params = []
    f1_1 = frame_exp(files_valid, y_valid, frame_size_range=frame_range, landmark_num=274, threshold=299.5, padding_size=5, colorspace_comp='R')
    opt_params = frame_range[np.argmax(f1_1)]
    f1_2 = landmark_exp(files_valid, y_valid, frame_size=frame_range[np.argmax(f1_1)], landmark_range_list=landmark_range, threshold=299.5, padding_size=5, colorspace_comp='R')
    opt_params = np.append(opt_params, landmark_range[np.argmax(f1_2)])
    f1_3 = threshold_exp(files_valid, y_valid, frame_size=frame_range[np.argmax(f1_1)], landmark_num=landmark_range[np.argmax(f1_2)], threshold_range_list=threshold_range, padding_size=5, colorspace_comp='R')
    opt_params = np.append(opt_params, threshold_range[np.argmax(f1_3)])
    f1_4 = padding_exp(files_valid, y_valid, frame_size=frame_range[np.argmax(f1_1)], landmark_num=landmark_range[np.argmax(f1_2)], threshold=threshold_range[np.argmax(f1_3)], padding_range_list=padding_range, colorspace_comp='R')
    opt_params = np.append(opt_params, padding_range[np.argmax(f1_4)])
    f1_5 = colorspace_exp(files_valid, y_valid, frame_size=frame_range[np.argmax(f1_1)], landmark_num=landmark_range[np.argmax(f1_2)], threshold=threshold_range[np.argmax(f1_3)], padding_size=padding_range[np.argmax(f1_4)], colorspace_list=colorspace_range)
    opt_cspace = colorspace_range[np.argmax(f1_5)]
    return opt_params, opt_cspace

def kfold(files_folder, y_true):
    # kf = KFold(n_splits=5, shuffle=True)
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf.get_n_splits(files_folder)
    # print(kf.split(files_folder))
    print(kf.split(files_folder, y_true))
    f1_score_folder = []
    # for train_index, test_index in kf.split(files_folder, y_true):
    for train_index, test_index in kf.split(files_folder, y_true):
        print("TRAIN:", train_index, "TEST:", test_index)
        files_folder = np.array(files_folder)
        X_train, X_test = files_folder[train_index], files_folder[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index]
        # frame_ranges = np.arange(5, 75, 5)
        frame_ranges = [50]
        landmark_ranges = np.arange(1, 512, 1)
        # landmark_ranges = [274]
        # threshold_ranges = np.arange(298, 302, 0.5)
        # threshold_ranges = np.arange(295, 305, 0.5)
        threshold_ranges = [299.5]
        # padding_ranges = np.arange(1, 10, 1)
        padding_ranges = [5]
        # colorspace_ranges = ['R', 'B', 'G', 'H', 'S', 'V', 'L', 'A', 'B_l', 'Y', 'Cr', 'Cb']
        colorspace_ranges = ['R']
        f1, color = validation(X_train, y_train, frame_ranges, landmark_ranges, threshold_ranges, padding_ranges, colorspace_ranges)
        print('Max F1 Score:', f1)
        print('Color:', color)
        f1_temp = folder_eval(file_list = X_test, y_true_list = y_test, frames=f1[0], landmark=f1[1], threshold = f1[2], padding = f1[3], colorspace_comp = color)
        f1_score_folder.append(f1_temp)
        # print(f1_score_folder)
        # print('X_train, X_test:', X_train, X_test)
    print(f1_score_folder)
    print(np.mean(f1_score_folder))
    print(np.std(f1_score_folder))
    return np.mean(f1_score_folder)
        
def main(experiment_status, experiment):
    if experiment_status == True:
        f1_rand = []
        for random_iter in tqdm(range(0, 1)):
            print('-'*40)
            print('Random Iteration:', random_iter)
            print('_'*40)
            files_random, y_true_random = randomizer(files, sample_size=100)
            # files_random = []
            # y_true_random = []

            # for rand_sample in range(0, 100):
            #     random_index = random.randrange(len(files))
            #     # y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
            #     y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
            #     y_true_random.append(y_true[random_index])
            #     files_random.append(files[random_index])
            # final_temps = []
            # f1_sc = []
            if experiment == 'Frame':
                frames_range = np.arange(5, 130, 15)
                f1_rand.append(frame_exp(file_list = files_random, y_true_list = y_true_random, frame_size_range = frames_range))
                x_values = frames_range

            if experiment == 'Landmark':
                landmarks_range = np.arange(1, 512, 1)
                f1_rand.append(landmark_exp(file_list = files_random, y_true_list = y_true_random, landmark_range_list = landmarks_range))
                x_values = landmarks_range
            
            if experiment == 'Threshold':
                thresholds_list = np.linspace(24, 32, 10)
                f1_rand.append(threshold_exp(file_list = files_random, y_true_list = y_true_random, threshold_range_list = thresholds_list))
                x_values = thresholds_list

            if experiment == 'Padding':
                padding_range = np.arange(1, 3, 1)
                f1_rand.append(padding_exp(file_list = files_random, y_true_list = y_true_random, padding_range_list = padding_range))
                x_values = padding_range

            if experiment == 'Colorspace':
                colorspace_range = ['B', 'G', 'H', 'S', 'V', 'L', 'A', 'B_l', 'Y', 'Cr', 'Cb']
                f1_rand.append(colorspace_exp(file_list = files_random, y_true_list = y_true_random, colorspace_list = colorspace_range))
                x_values = colorspace_range

                

        print(f1_rand)
        print(np.shape(f1_rand))
        f1_rand = np.asarray(f1_rand)
        f1_mean = []
        f1_std = []
        for i in range(0, np.shape(f1_rand)[1]):
            f1_mean.append(np.mean(f1_rand[:, i]))
            f1_std.append(np.std(f1_rand[:, i]))
        print(f1_mean, f1_std)
        # plt.bar(frames_x, f1_mean, width=0.2, yerr = f1_std, color = 'orange', ecolor = 'brown', capsize = 2)
        plt.bar(x_values, f1_mean, width=0.2, yerr = f1_std, color = 'orange', ecolor = 'brown', capsize = 2)
        plt.savefig("Landmarks_frames.png")
        # plt.show()
        # plt.close()
    
    elif experiment_status == False:
        # y_true = pd.read_csv('Ground_Truth.csv', sep='delimiter', header=None)
        # y_true = np.squeeze(np.asarray(y_true))

        y_true = [1]

        # print(y_true)
        # print(np.shape(y_true))
        # y_true = [1]*len(files)
        # y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        # landmarkarray = np.squeeze(np.arange(1, 512, 1))
        f1_score_folder = folder_eval(file_list = files, y_true_list = y_true, frames=100, landmark=9, threshold=300.5, padding=5, colorspace_comp='B')
        # print(f1_score_folder[1])
        return f1_score_folder[1]
    
    elif experiment_status == 'Valid':
        # y_true = pd.read_csv('Ground_Truth.csv', header=None)
        # y_true = np.squeeze(np.asarray(y_true))
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        validscore = kfold(files, y_true=y_true)
        print(validscore)
        

        # print(f1_rand)
        # print(np.shape(f1_rand))
        # f1_rand = np.asarray(f1_rand)
        # f1_mean = []
        # f1_std = []
        # for i in range(0, np.shape(f1_rand)[1]):
        #     f1_mean.append(np.mean(f1_rand[:, i]))
        #     f1_std.append(np.std(f1_rand[:, i]))
        # print(f1_mean, f1_std)
        # # plt.bar(frames_x, f1_mean, width=0.2, yerr = f1_std, color = 'orange', ecolor = 'brown', capsize = 2)
        # plt.bar(x_values, f1_mean, width=0.2, yerr = f1_std, color = 'orange', ecolor = 'brown', capsize = 2)
        # plt.savefig("Landmarks_frames.png")
        # plt.show()
        # plt.close()



# main(experiment_status = False, experiment='Landmark')