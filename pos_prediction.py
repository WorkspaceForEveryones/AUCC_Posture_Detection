import os
import cv2
import mediapipe as mp
import math

def cal_angle(start1 , start2 , end1 , end2):
    x1 , y1 = start1.x , start1.y
    x2 , y2 = end1.x , end1.y
    x3 , y3 = start2.x , start2.y
    x4 , y4 = end2.x , end2.y
    
    cos_angle = ((x2 - x1) * (x4 - x3)) + ((y4 - y3) * (y2 - y1))
    cos_angle /= math.sqrt((((y2 - y1)**2) + ((x2 - x1)**2)) * (((y4 - y3)**2) + ((x4 - x3)**2)))

    angle = math.acos(cos_angle) * (180 / math.pi)
    return angle

def check_pose (image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    posture1 = posture_1(mp_pose,results)
    posture2 = posture_2(mp_pose,results)
    posture3 = posture_3(mp_pose,results)
    posture4 = posture_4(mp_pose,results)
    if posture4:
        return 4
    elif (posture3 and posture2) or posture3:
        return 3
    elif (posture2 and posture1) or posture2:
        return 2
    elif posture1:
        return 1
    else: return 0
    
def posture_1(mp_pose , results):
    # Define the landmark pairs
    hip_knee_pairs = [(mp_pose.PoseLandmark.LEFT_KNEE , mp_pose.PoseLandmark.LEFT_HIP),
                    (mp_pose.PoseLandmark.RIGHT_KNEE , mp_pose.PoseLandmark.RIGHT_HIP)]
    knee_ankle_pairs = [(mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)]

    left_point = []
    for pair in hip_knee_pairs:
        start1 = results.pose_landmarks.landmark[pair[0]]
        end1 = results.pose_landmarks.landmark[pair[1]]
        left_point.append((start1,end1))

    right_point = []
    for pair in knee_ankle_pairs:
        start2 = results.pose_landmarks.landmark[pair[0]]
        end2 = results.pose_landmarks.landmark[pair[1]]
        right_point.append((start2,end2))
        
    angle1 = cal_angle(left_point[0][0],right_point[0][0],left_point[0][1],right_point[0][1])
    angle2 = cal_angle(right_point[1][0],left_point[1][0],right_point[1][1],left_point[1][1])

    diff_angle_knee = abs(angle1-angle2)
    if diff_angle_knee <= 50 and diff_angle_knee >= 12:
        return True
    else:
        return False
     
def posture_2(mp_pose , results):
    # Define the list of landmark pairs to connect with lines
    landmark_pairs = [(mp_pose.PoseLandmark.RIGHT_HIP ,mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE)]
    landmark_pairs1 = [(mp_pose.PoseLandmark.RIGHT_HIP ,mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE)]
    
    start1 = results.pose_landmarks.landmark[landmark_pairs[0][0]] #(x1, y1)
    end1 = results.pose_landmarks.landmark[landmark_pairs[0][1]] #(x2, y2)
    start2 = results.pose_landmarks.landmark[landmark_pairs[1][0]] #(x3,y3)
    end2 = results.pose_landmarks.landmark[landmark_pairs[1][1]] #(x4,y4)
    
    angle_right = cal_angle(start1 , start2 ,end1 ,end2)
    
    start1 = results.pose_landmarks.landmark[landmark_pairs1[0][0]]
    end1 = results.pose_landmarks.landmark[landmark_pairs1[0][1]]
    start2 = results.pose_landmarks.landmark[landmark_pairs1[1][0]]
    end2 = results.pose_landmarks.landmark[landmark_pairs1[1][1]]
   
    angle_left = cal_angle(start1 , start2 ,end1 ,end2)
    
    if (angle_left <= 80 and angle_left >= 45) or (angle_right <= 80 and angle_right >= 45):
        return True
    else: return False
        
def posture_3(mp_pose , results):
    # Define the list of landmark pairs to connect with lines
    landmark_pairs_right = [(mp_pose.PoseLandmark.RIGHT_SHOULDER , mp_pose.PoseLandmark.RIGHT_HIP),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST)]
    landmark_pairs_left = [(mp_pose.PoseLandmark.LEFT_SHOULDER , mp_pose.PoseLandmark.LEFT_HIP),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST)]
    
    start1 = results.pose_landmarks.landmark[landmark_pairs_right[0][0]]
    end1 = results.pose_landmarks.landmark[landmark_pairs_right[0][1]]
    start2 = results.pose_landmarks.landmark[landmark_pairs_right[1][0]]
    end2 = results.pose_landmarks.landmark[landmark_pairs_right[1][1]]
    
    angle_right = cal_angle(start1 , start2 ,end1 ,end2)
    
    start1 = results.pose_landmarks.landmark[landmark_pairs_left[0][0]]
    end1 = results.pose_landmarks.landmark[landmark_pairs_left[0][1]]
    start2 = results.pose_landmarks.landmark[landmark_pairs_left[1][0]]
    end2 = results.pose_landmarks.landmark[landmark_pairs_left[1][1]]
    
    angle_left = cal_angle(start1 , start2 ,end1 ,end2)
    
    if (angle_left <= 45 and angle_left >= 20) or (angle_right <= 45 and angle_right >= 20):
        return True
    else: return False

def posture_4(mp_pose , results):
    # Define the list of landmark pairs to connect with lines
    landmark_pairs = [(mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    (mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.RIGHT_ANKLE)]
    landmark_pairs1 = [(mp_pose.PoseLandmark.LEFT_HIP,
                    mp_pose.PoseLandmark.LEFT_SHOULDER),
                    (mp_pose.PoseLandmark.LEFT_HIP,
                    mp_pose.PoseLandmark.LEFT_ANKLE)]
    
    start1 = results.pose_landmarks.landmark[landmark_pairs[0][0]]
    end1 = results.pose_landmarks.landmark[landmark_pairs[0][1]]
    start2 = results.pose_landmarks.landmark[landmark_pairs[1][0]]
    end2 = results.pose_landmarks.landmark[landmark_pairs[1][1]]
    # Check if the angle is less than 40 degrees and  more than 10 degrees print "Yes" or "No"
    angle_right = cal_angle(start1 , start2 ,end1 ,end2)
    start1 = results.pose_landmarks.landmark[landmark_pairs1[0][0]]
    end1 = results.pose_landmarks.landmark[landmark_pairs1[0][1]]
    start2 = results.pose_landmarks.landmark[landmark_pairs1[1][0]]
    end2 = results.pose_landmarks.landmark[landmark_pairs1[1][1]]
    # Check if the angle is less than 40 degrees and  more than 10 degrees print "Yes" or "No"
    angle_left = cal_angle(start1 , start2 ,end1 ,end2)

    # Check if the different of angle is less than -- degrees and print "Yes" or "No"
    if (angle_right >= 170 and angle_right <= 190) or (angle_left >= 170 and angle_left <= 190):
        return True
    else:
        return False
        
def process_images_and_save_info(folder_path, output_file):
    image_files = sorted([os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith((".jpg", ".png"))])
    with open(output_file, 'w') as file:
        for image_path in image_files:
            posture = check_pose(image_path)
            file.write(f"{posture}\n")

if __name__ == "__main__":
    # folder_path = 'new_dataset/new_image'
    # output_file = 'new_dataset/predict_postures.txt'
    folder_path = 'dada_dataset/dada_image'
    output_file = 'dada_dataset/predict_postures.txt'
    process_images_and_save_info(folder_path, output_file)
