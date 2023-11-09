import mediapipe as mp
import deepLabCut as dlc
import openpose as op
import numpy as np

#ใช้ library 
def detect_coordinates():
    x_coordinates = np.array([])#น่าจะเป็นพิกัด xของท่า
    y_coordinates = np.array([])#น่าจะเป็นพิกัด yของท่า
    return x_coordinates, y_coordinates

#เปรียบเทียบค่าที่ได้กับ correctCoord.txt
def compare_coordinates(x_detected, y_detected, correct_coord_file):
    #อ่านไฟล์ correctCoord.txt
    with open(correct_coord_file, 'r') as file:
        correct_coords = file.readlines()

    # แปลงข้อมูลจากไฟล์เป็นรายการของพิกัด x และ y
    correct_x = [float(line.split()[0]) for line in correct_coords] #แปลงค่า x ที่อยู่ในไฟล์text
    correct_y = [float(line.split()[1]) for line in correct_coords] #แปลงค่า y ที่อยู่ในไฟล์text

    # เปรียบเทียบข้อมูลที่ได้กับข้อมูลถูกต้อง
    diff_x = x_detected - np.array(correct_x)
    diff_y = y_detected - np.array(correct_y)

    return diff_x, diff_y

if __name__ == "__main__":
    x_detected, y_detected = detect_coordinates()
    correct_coord_file = "correctCoord.txt"
    diff_x, diff_y = compare_coordinates(x_detected, y_detected, correct_coord_file)

    # วนลูปค่า x y ที่ได้มา
    for i in range(len(diff_x)):
        print(f"ความต่าง x ที่ได้ที่จุดที่ {i + 1}: {diff_x[i]}")
        print(f"ความต่าง y ที่ได้ที่จุดที่ {i + 1}: {diff_y[i]}")
