import mediapipe as mp
import openpose as op
import numpy as np
import math
def detect_coordinates():
    x_coordinates = np.array([])  # น่าจะเป็นพิกัด x ของท่า
    y_coordinates = np.array([])  # น่าจะเป็นพิกัด y ของท่า
    return x_coordinates, y_coordinates

# เปรียบเทียบค่าที่ได้กับ correctCoord.txt
def compare_coordinates(correct_coord_file):
    # อ่านไฟล์ correctCoord.txt
    with open(correct_coord_file, 'r') as file:
        correct_coords = file.readlines()

    # แปลงข้อมูลจากไฟล์เป็นรายการของพิกัด x และ y
    correct_x = [float(line.split()[0]) for line in correct_coords]  # แปลงค่า x ที่อยู่ในไฟล์ text
    correct_y = [float(line.split()[1]) for line in correct_coords]  # แปลงค่า y ที่อยู่ในไฟล์ text

    correct_x = np.array(correct_x)
    correct_y = np.array(correct_y)

    return correct_x, correct_y

def mean_absolute_error(compare_coordinates, detect_coordinates):
    n = len(compare_coordinates)
    if n != len(detect_coordinates):
        raise ValueError("จำนวนตัวอย่างไม่เท่ากัน")

    mae = sum(abs(compare_coordinates[i] - detect_coordinates[i]) for i in range(n)) / n
    return mae

def root_mean_squared_error(compare_coordinates, detect_coordinates):
    n = len(compare_coordinates)
    if n != len(detect_coordinates):
        raise ValueError("จำนวนตัวอย่างไม่เท่ากัน")

    squared_errors = [(compare_coordinates[i] - detect_coordinates[i]) ** 2 for i in range(n)]
    mean_squared_error = sum(squared_errors) / n
    rmse = math.sqrt(mean_squared_error)
    return rmse

if __name__ == "__main__":
    x_detected, y_detected = detect_coordinates()
    correct_coord_file = "correctCoord.txt"
    correct_x, correct_y = compare_coordinates(correct_coord_file)

    mae_result = mean_absolute_error(correct_x, x_detected)
    print(f"Mean Absolute Error: {mae_result}")
