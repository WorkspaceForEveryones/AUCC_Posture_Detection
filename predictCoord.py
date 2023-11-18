import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp
import math

class PointRecorder:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_paths = sorted([os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith((".jpg", ".png"))])
        self.points_data = {}
        self.current_image = None
        self.counter = 0
        self.not_good = []

    def load_image(self, image_path):
        self.current_image = image_path
        if self.current_image not in self.points_data:
            self.points_data[self.current_image] = []
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # Initialize MediaPipe Pose model
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

        # Load the input image
        image = cv2.imread(image_path)

        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose
        results = pose.process(image)

        # Draw landmarks on the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        height, width, _ = image.shape
        try:
            right_wrist_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width
            right_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height
            right_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width
            right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height
            right_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * width
            right_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * height
            right_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * width
            right_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * height
            right_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width
            right_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height
            left_wrist_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width
            left_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height
            left_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width
            left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height
            left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * width
            left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * height
            left_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * width
            left_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * height
            left_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * width
            left_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * height

            self.points_data[self.current_image].append((right_wrist_x, right_wrist_y))
            self.points_data[self.current_image].append((right_shoulder_x, right_shoulder_y))
            self.points_data[self.current_image].append((right_hip_x, right_hip_y))
            self.points_data[self.current_image].append((right_knee_x, right_knee_y))
            self.points_data[self.current_image].append((right_ankle_x, right_ankle_y))
            self.points_data[self.current_image].append((left_wrist_x, left_wrist_y))
            self.points_data[self.current_image].append((left_shoulder_x, left_shoulder_y))
            self.points_data[self.current_image].append((left_hip_x, left_hip_y))
            self.points_data[self.current_image].append((left_knee_x, left_knee_y))
            self.points_data[self.current_image].append((left_ankle_x, left_ankle_y))
        except:
            self.not_good.append(image_path)


    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            for image, points in self.points_data.items():
                f.write(', '.join([f'{point[0]}, {point[1]}' for point in points]))
                f.write('\n')
        print(self.not_good)

    def show_plot(self):
        plt.show()

if __name__ == "__main__":
    image_folder = 'dada_dataset/dada_image'
    output_file = 'dada_dataset/predict_coord_file.txt'

    recorder = PointRecorder(image_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            recorder.load_image(image_path)
            recorder.show_plot()

    recorder.save_to_file(output_file)
    print(f"Points recorded to {output_file}")
