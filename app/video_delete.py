import os

def delete_except(directory, keep_filename):
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and file != keep_filename:
            os.remove(file_path)
            print(f"Deleted: {file_path}")