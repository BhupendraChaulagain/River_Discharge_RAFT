import cv2
import time
import os
from dotenv import load_dotenv

VELOCITY_DATA_DIR = "velocity_data"
load_dotenv()
# RTSP URL with authentication
rtsp_url = os.getenv("RTSP_URL")

# Captured video duration
video_duration = 2

# Set the interval of 10 minutes
capture_interval = 70 

# Output directory for captured videos
output_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(output_dir, exist_ok=True)

# Target resolution for output video (e.g., 640x360)
target_resolution = (640, 360)

# Initialization of video capture
is_capturing = None

def initialize_capture():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream. Check credentials or connection.")
        return None
    return cap

# Flushing frames from buffer
def flush_buffer(cap, frames_to_flush=10):
    for _ in range(frames_to_flush):
        cap.grab()

# Capturing video
def capture_video(video_index):
    
    # Initializing capture
    cap = initialize_capture()
    if not cap:
        print(f"Failed to initialize capture for video {video_index}")
        return False

    flush_buffer(cap)

    frame_width = target_resolution[0]
    frame_height = target_resolution[1]
    print(f"Capturing video {video_index} with target dimensions: {frame_width}x{frame_height}")

    output_filename = os.path.join(output_dir, f"video_{video_index}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

    start_time = time.time()
    while time.time() - start_time < video_duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Retrying...")
            cap = cv2.VideoCapture(rtsp_url)
            time.sleep(0.1)  # Short pause before retrying
            continue

        # Resize frame to target resolution
        resized_frame = cv2.resize(frame, target_resolution)
        out.write(resized_frame)

    out.release()
    cap.release()
    print(f"Video {video_index} saved as {output_filename}")
    return True

# Infinite loop to capture videos continuously
def start_capture():
    global is_capturing
    is_capturing = True
    video_index = 1
    while is_capturing:
        print(f"Starting capture for video {video_index}...")
        success = capture_video(video_index)
        if not success:
            print("Skipping this capture due to errors.")
        video_index += 1
        print(f"Waiting for {capture_interval} seconds before the next capture...")
        time.sleep(capture_interval)

def stop_capture():
    global is_capturing
    is_capturing = False
    print("Stopping the video capture.")
