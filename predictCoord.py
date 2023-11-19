import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp
import math
import argparse

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

    def load_image_openpose(self, image_path):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
        parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
        parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
        parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

        args = parser.parse_args()

        inWidth = args.width
        inHeight = args.height

        cap = cv2.VideoCapture(image_path)

        hasFrame, frame = cap.read()

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
        self.current_image = image_path
        if self.current_image not in self.points_data:
            self.points_data[self.current_image] = []
        points = []
        assert(19 == out.shape[1])
        for i in range(19):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((x, y))
        try:
            self.points_data[self.current_image].append((points[4][0], points[4][1]))
            self.points_data[self.current_image].append((points[2][0], points[2][1]))
            self.points_data[self.current_image].append((points[8][0], points[8][1]))
            self.points_data[self.current_image].append((points[9][0], points[9][1]))
            self.points_data[self.current_image].append((points[10][0], points[10][1]))
            self.points_data[self.current_image].append((points[7][0], points[7][1]))
            self.points_data[self.current_image].append((points[5][0], points[5][1]))
            self.points_data[self.current_image].append((points[11][0], points[11][1]))
            self.points_data[self.current_image].append((points[12][0], points[12][1]))
            self.points_data[self.current_image].append((points[13][0], points[13][1]))
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
    image_folder = 'new_dataset/new_image'
    output_file = 'new_dataset/predict_coord_openpose_file.txt'

    recorder = PointRecorder(image_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            recorder.load_image_openpose(image_path)
            recorder.show_plot()

    recorder.save_to_file(output_file)
    print(f"Points recorded to {output_file}")
