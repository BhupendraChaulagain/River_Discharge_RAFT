import os
from app.frame_extraction import extract_frames_by_time


def process_multiple_videos(input_dir, output_dir, start_time, end_time, frame_count=2):
    # Get all video files from the input directory
    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    
    if not video_files:
        raise ValueError("No video files found in the input directory.")

    print(f"Found {len(video_files)} video(s) in {input_dir}.")
    print("FIne till here")


    all_extracted_frames = {}
    for video_path in video_files:
        try:
            print(f"Processing video: {video_path}")
            extracted_frames = extract_frames_by_time(
                video_path,
                output_dir,
                start_time,
                end_time,
                frame_count
            )
            all_extracted_frames[video_path] = extracted_frames
        except ValueError as e:
            print(f"Error processing {video_path}: {e}")
    
    return all_extracted_frames