import os
import shutil

def delete_all_files_in_directory(directory: str):
   
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete subdirectory
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
