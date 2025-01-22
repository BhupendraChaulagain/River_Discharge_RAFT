import os
import time

def video_save(video_path, wait_time=5):
    
    if not os.path.exists(video_path):
        print(f"File does not exist: {video_path}")
        return False

    last_size = -1
    elapsed_time = 0

    while elapsed_time < wait_time:
        try:
            current_size = os.path.getsize(video_path)
            if current_size > 0 and current_size == last_size:
                print(f"File size stabilized: {current_size} bytes")
                return True  # File writing is complete
            last_size = current_size
        except Exception as e:
            print(f"Error checking file size: {e}")
            return False
        
        time.sleep(1)  # Wait before checking again
        elapsed_time += 1

    print(f"File size did not stabilize within {wait_time} seconds")
    return False