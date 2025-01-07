import cv2

def save_arrowed_image(image, output_path):
    
    try:
        cv2.imwrite(output_path, image)
    except Exception as e:
        raise RuntimeError(f"Failed to save the image: {str(e)}")