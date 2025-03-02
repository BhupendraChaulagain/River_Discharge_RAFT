import cv2
import os



def extract_frames_by_time(video_path, output_dir, start_time, end_time, frame_count=5, fps=None):
   
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)

    
    if fps is None:
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    
    # Validate start and end times
    if start_time >= duration or end_time > duration or start_time >= end_time:
        raise ValueError(f"Invalid start or end time: Start = {start_time}, End = {end_time}, Duration = {duration:.2f} seconds")

    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames)  # Ensuring don't exceed total frames
    available_frames = end_frame - start_frame

    # Calculate the actual number of frames to extract
    frame_count = min(frame_count, available_frames)

    

    extracted_frames = []

    

    # Extract frames
    for i in range(frame_count):
        frame_no = start_frame + i 
        if frame_no >= end_frame:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_no}.")
            break

        frame_filename = os.path.join(output_dir, f"{video_name}_frame_{i}.jpg")
        cv2.imwrite(frame_filename, frame)
        extracted_frames.append(frame_filename)
        

    cap.release()

    if not extracted_frames:
        raise ValueError("No frames were extracted.")
    
    return extracted_frames
