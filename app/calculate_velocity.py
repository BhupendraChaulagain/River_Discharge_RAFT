import torch
import argparse
import cv2
import numpy as np
import pandas as pd
import os
import sys
from collections import deque
from app.delete_frames import delete_all_files_in_directory
from app.clear_create_dir import clear_and_create_directory
from app.video_delete import delete_except

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'core')))
# Import RAFT components
from core.raft import RAFT
from core.utils.utils import InputPadder

UPLOAD_DIR = "uploads"


def load_model(model_path):
    """Load the pre-trained RAFT model."""
    args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model = model.module
    model.to('cpu')
    model.eval()
    return model

def preprocess_frame(frame, scale_factor=0.25):
    #preprocessing input
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    original_h, original_w = frame.shape[:2]
    
    resized_h, resized_w = int(original_h * scale_factor), int(original_w * scale_factor)
   
    resized_frame = cv2.resize(frame, (resized_w, resized_h))
    frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float()
    return frame_tensor, (original_w, original_h), (resized_w, resized_h)

def calculate_velocity(frame_folder, output_csv, model_path, x_range=(0, 640), y_range=(0, 480), 
                       num_segments=4, frame_step=1, scale_factor=0.25):
    
    model = load_model(model_path)

    #list of frame files
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png') or f.endswith('.jpg')])

    print(f"Number of frames found: {len(frame_files)}")
    if len(frame_files) < 2:
        print("Error: Not enough frames to calculate optical flow.")
        return []
    
    velocity_data = []

    # Width of section calculated
    x_start, x_end = x_range
    y_start, y_end = y_range
    x_step = (x_end - x_start) // num_segments

    segment_width = (x_end - x_start) / num_segments
    segment_boundaries = [(x_start + i * segment_width, x_start + (i + 1) * segment_width) 
                         for i in range(num_segments)]
    
    # Calculation of optical flow
    for i in range(len(frame_files) - frame_step):
        # Load consecutive frames
        prev_frame_path = os.path.join(frame_folder, frame_files[i])
        curr_frame_path = os.path.join(frame_folder, frame_files[i + frame_step])

        prev_frame = cv2.imread(prev_frame_path)
        curr_frame = cv2.imread(curr_frame_path)

        if prev_frame is None or curr_frame is None:
            print(f"Error loading frames: {prev_frame_path} or {curr_frame_path}")
            continue

        # Preprocessing frames and get dimensions
        prev_frame, original_dims, resized_dims = preprocess_frame(prev_frame, scale_factor)
        curr_frame, _, _ = preprocess_frame(curr_frame, scale_factor)

        padder = InputPadder(prev_frame.shape)
        prev_frame, curr_frame = padder.pad(prev_frame, curr_frame)

        # Calculating optical flow with RAFT
        with torch.no_grad():
            flow = model(prev_frame, curr_frame)[1]
            flow = flow[0].numpy().transpose(1, 2, 0)


        # Apply noise thresholding
        
        

        original_w, original_h = original_dims
        resized_w, resized_h = resized_dims
        scale_x = original_w / resized_w
        scale_y = original_h / resized_h

        
        scaled_x_start = int(x_start / scale_x)
        scaled_x_end = int(x_end / scale_x)
        scaled_y_start = int(y_start / scale_y)
        scaled_y_end = int(y_end / scale_y)

        
        for y in range(int(scaled_y_start), int(scaled_y_end)):
            
            for x in range(int(scaled_x_start),int(scaled_x_end)):
                if x >= flow.shape[1] or y >= flow.shape[0]:
                    continue

                flow_at_point = flow[int(y), int(x)]
                velocity_x, velocity_y = flow_at_point[0], flow_at_point[1]

                # Scaling the coordinates and velocity back to original size
                original_x = int(x * scale_x)
                original_y = int(y * scale_y)

               
                for segment_num, (seg_start, seg_end) in enumerate(segment_boundaries, 1):
                    if seg_start <= original_x < seg_end:
                        velocity_data.append({
                            'frame': i,
                            'segment': segment_num,
                            'x-coordinate': original_x,
                            'y-coordinate': original_y,
                            'velocity_x': velocity_x * scale_x,
                            'velocity_y': velocity_y * scale_y
                        })
                        break

    
    delete_all_files_in_directory(frame_folder)
    
    
    output_directory = os.path.dirname(output_csv)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if velocity_data:
        df = pd.DataFrame(velocity_data)
        df.to_csv(output_csv, index=False)
        
    else:
        print("No velocity data to save.")

    delete_except(UPLOAD_DIR, "video_1.avi")
    return velocity_data
