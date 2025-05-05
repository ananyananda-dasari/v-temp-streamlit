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


matfilename = r'./Mat_Files/N14_Lab_test_fit.mat'
mat = scipy.io.loadmat(matfilename)
# print(np.shape(mat))
# for key, value in mat.items():
#     print(f"{key}: {value}")

mat = {k:v for k, v in mat.items() if k[0] != '_'}
data_2d = pd.DataFrame({k: v[0] for k, v in mat.items()})
# print(np.shape(mat['fit']))
print(np.shape(data_2d))
print(np.shape(data_2d['fit']))

print(np.shape(data_2d['fit'][15][3][1][1]))

print(data_2d['fit'][15][3][1][1])

print(data_2d['fit'][15][4][0][1])

# print(data_2d['fit'][1])
# print(data_2d['fit'][1][2])