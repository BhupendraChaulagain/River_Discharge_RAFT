import torch
import argparse
import cv2
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'core')))
from core.raft import RAFT
from core.utils.utils import InputPadder

def load_model(model_path):
    args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.module
    model.to('cpu')
    model.eval()
    return model

def preprocess_frame(frame, size):
    frame = cv2.resize(frame, size)
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    return frame

def main():
    model_path = 'raft-sintel.pth'  # Update with the path to your model
    frame1_path = 'frame_0.jpg'  # Update with the path to your first image
    frame2_path = 'frame_1.jpg'  # Update with the path to your second image

    model = load_model(model_path)

    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    original_height, original_width = frame1.shape[:2]
    new_height, new_width = original_height // 4, original_width // 4

    frame1 = preprocess_frame(frame1, (new_width, new_height))
    frame2 = preprocess_frame(frame2, (new_width, new_height))

    frame1 = frame1.cpu()
    frame2 = frame2.cpu()

    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)

    with torch.no_grad():
        flow = model(frame1, frame2)[1]
        flow = flow[0].numpy().transpose(1, 2, 0)

    indices = [(i, j) for i in range(0, flow.shape[0], 2) for j in range(0, flow.shape[1], 2)]
    velocity_data = [(i * 2, j * 2, flow[i, j, 0], flow[i, j, 1]) for i, j in indices]

    df = pd.DataFrame(velocity_data, columns=['x', 'y', 'vx', 'vy'])
    df.to_csv('velocity_data.csv', index=False)  # Update with your desired save path

    print('Velocity data saved to velocity_data.csv')

if __name__ == "__main__":
    main()
