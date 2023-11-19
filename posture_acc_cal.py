import cv2
import mediapipe as mp
import math  
import numpy as np  
from sklearn.metrics import mean_absolute_error 

correct = open("/Volumes/Mill/Mac_Mini/AUCC_Posture_Detection/new_dataset/labels.txt", "r")
predict = open("/Volumes/Mill/Mac_Mini/AUCC_Posture_Detection/new_dataset/predict_posture.txt", "r")
# Example usage:
rmse_value = 0
mae_value = 0
pos1_cor = 0
pos1_pre = 0
pos2_cor = 0
pos2_pre = 0
pos3_cor = 0
pos3_pre = 0
pos4_cor = 0
pos4_pre = 0
pos_wrong_cor = 0
pos_wrong_pre = 0
for i in correct:
    f1 = int(i)
    f2 = int(predict.readline())

    if f1 == f2:
        if f1 == 1: 
            pos1_cor += 1
            pos1_pre += 1
        if f1 == 2: 
            pos2_cor += 1
            pos2_pre += 1
        if f1 == 3: 
            pos3_cor += 1
            pos3_pre += 1
        if f1 == 4: 
            pos4_cor += 1
            pos4_pre += 1
        if f1 == 0: 
            pos_wrong_cor += 1
            pos_wrong_pre += 1
    else:
        if f1 == 1: 
            pos1_cor += 1
        if f1 == 2: 
            pos2_cor += 1
        if f1 == 3: 
            pos3_cor += 1
        if f1 == 4: 
            pos4_cor += 1
        if f1 == 0: 
            pos_wrong_cor += 1

print("posture 1:", pos1_pre * 100 / pos1_cor)
print("posture 2:", pos2_pre * 100 / pos2_cor)
print("posture 3:", pos3_pre * 100 / pos3_cor)
print("posture 4:", pos4_pre * 100 / pos4_cor)
print("wrong posture:", pos_wrong_pre * 100 / pos_wrong_cor)

