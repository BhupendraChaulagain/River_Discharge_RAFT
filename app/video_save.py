import os
import time

def video_save(video_path, wait_time=5):
    
    last_size = -1
    elapsed_time = 0
    while elapsed_time < wait_time:
        current_size = os.path.getsize(video_path)
        if current_size == last_size:
            return True  # Video is fully written
        last_size = current_size
        time.sleep(1)  # Wait for 1 second before checking again
        elapsed_time += 1
    return False