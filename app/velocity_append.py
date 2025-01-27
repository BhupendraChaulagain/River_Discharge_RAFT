import pandas as pd
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np

class VelocityDataHandler(FileSystemEventHandler):
    def __init__(self, velocity_data_dir, output_file, num_segments, scaling_factor, fps):
        self.velocity_data_dir = velocity_data_dir
        self.output_file = output_file
        self.num_segments = num_segments
        self.scaling_factor = scaling_factor
        self.fps = fps
        self.processed_files = set()
        
    def process_velocity_file(self, velocity_file, scaling_factor, fps):
        try:
            # Read the velocity data
            df = pd.read_csv(velocity_file)
            
            # Calculate segment averages
            segment_avg_velocity_x = df.groupby('segment')['velocity_x'].mean()
            segment_avg_velocity_y = df.groupby('segment')['velocity_y'].mean()
            
            # Calculate total velocity for each segment
            segment_velocities = {
                segment: (np.sqrt(segment_avg_velocity_x[segment]**2 + segment_avg_velocity_y[segment]**2))*scaling_factor*fps
                for segment in range(1, self.num_segments + 1)
            }
            
            # Create a row for the consolidated data
            video_name = os.path.basename(velocity_file).replace('velocity_data_', '').replace('.csv', '')
            row_data = {'video_name': f'video_{video_name}'}
            
            # Add segment velocities
            for segment, velocity in segment_velocities.items():
                row_data[f'segment_{segment}'] = velocity
                
            # Calculate average velocity across all segments
            row_data['average_velocity'] = np.mean(list(segment_velocities.values()))
            
            # Create or append to the consolidated CSV
            if os.path.exists(self.output_file):
                consolidated_df = pd.read_csv(self.output_file)
                consolidated_df = pd.concat([consolidated_df, pd.DataFrame([row_data])], ignore_index=True)
            else:
                consolidated_df = pd.DataFrame([row_data])
            
            consolidated_df.to_csv(self.output_file, index=False)
            print(f"Processed and added data from {velocity_file} to {self.output_file}")
            
        except Exception as e:
            print(f"Error processing {velocity_file}: {str(e)}")
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            if event.src_path not in self.processed_files and 'velocity_data_' in os.path.basename(event.src_path):
                # Wait briefly to ensure file is completely written
                time.sleep(2)
                self.process_velocity_file(event.src_path)
                self.processed_files.add(event.src_path)

def start_velocity_monitoring(velocity_data_dir, num_segments):
   
    output_file = os.path.join(velocity_data_dir, 'velocity.csv')
    event_handler = VelocityDataHandler(velocity_data_dir, output_file, num_segments)
    
    # Process any existing files first
    for file in os.listdir(velocity_data_dir):
        if file.startswith('velocity_data_') and file.endswith('.csv'):
            filepath = os.path.join(velocity_data_dir, file)
            if filepath not in event_handler.processed_files:
                event_handler.process_velocity_file(filepath)
                event_handler.processed_files.add(filepath)
    
    # Start monitoring for new files
    observer = Observer()
    observer.schedule(event_handler, velocity_data_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

