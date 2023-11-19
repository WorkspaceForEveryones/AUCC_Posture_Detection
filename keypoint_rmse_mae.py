import cv2
import mediapipe as mp
import math  
import numpy as np  
from sklearn.metrics import mean_absolute_error 

def calculate_rmse(coord_set1, coord_set2):
    if len(coord_set1) != len(coord_set2):
        raise ValueError("Coordinate sets must have the same length")

    # # Calculate squared differences for each pair of coordinates
    # squared_diff = [(x1 - x2)**2 + (y1 - y2)**2 for (x1, y1), (x2, y2) in zip(coord_set1, coord_set2)]

    # # Calculate the mean of the squared differences
    # mean_squared_diff = sum(squared_diff) / len(coord_set1)

    # # Take the square root to get the RMSE
    # rmse = math.sqrt(mean_squared_diff)

    MSE = np.square(np.subtract(coord_set1,coord_set2)).mean()   
   
    rmse = math.sqrt(MSE)  

    return rmse


correct = open("/Volumes/Mill/Mac_Mini/AUCC_Posture_Detection/new_dataset/correct_coord_file.txt", "r")
predict = open("/Volumes/Mill/Mac_Mini/AUCC_Posture_Detection/new_dataset/predict_coord_file.txt", "r")
# Example usage:
rmse_value = 0
mae_value = 0
for i in correct:
    coord1 = [float(x) for x in i.split(',')]
    coordinates1 = []
    for j in range(math.floor(len(coord1) / 4)):
        coordinates1.append([coord1[j*2], coord1[j*2 + 1]])

    coord2 = [float(x) for x in predict.readline().split(',')]
    coordinates2 = []
    for i in range(math.floor(len(coord2) / 4)):
        coordinates2.append([coord2[i*2], coord2[i*2 + 1]])

    rmse_value = (rmse_value + calculate_rmse(coord1, coord2)) / 2
    mae_value = (mae_value + mean_absolute_error(coord1, coord2)) / 2


print(f"RMSE: {rmse_value}")
print(f"MAE: {mae_value}")

