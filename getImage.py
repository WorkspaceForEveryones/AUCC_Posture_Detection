import cv2
import time

def capture_frames(video_path, output_folder):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture object is successfully opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval in seconds to capture one frame every 0.25 seconds
    interval = 0.25 / fps

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Start capturing frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame is successfully read
        if not ret:
            break  # Break the loop if the video ends

        # Save the frame to a file in the specified output folder
        timestamp = int(time.time() * 1000)  # Use milliseconds for better uniqueness
        image_path = os.path.join(output_folder, f'frame_{timestamp}.jpg')
        cv2.imwrite(image_path, frame)

        # Wait for the specified interval before capturing the next frame
        time.sleep(interval)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()

if __name__ == "__main__":
    import os

    # Replace 'your_video_file.mp4' with the path to your video file
    video_path = 'new_video.mp4'

    # Replace 'output_folder' with the path to the folder where you want to save the images
    output_folder = 'new_image'

    capture_frames(video_path, output_folder)
