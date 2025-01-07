from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse  
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import re
import shutil
from app.frame_extraction import extract_frames_by_time
from app.calculate_velocity import calculate_velocity
import cv2

import numpy as np
from pathlib import Path
import pandas as pd
from app.arrow_saved import save_arrowed_image
from app.arrows import draw_velocity_arrows_based_on_segments
from app.scale import calculate_scale
from starlette.middleware.sessions import SessionMiddleware
from app.delete_frames import delete_all_files_in_directory


# Initialize app and directories
app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")
# templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
app.mount("/rectangle", StaticFiles(directory="rectangle"), name="rectangle")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")



app.add_middleware(
    SessionMiddleware,
    secret_key="s20vfc45ki",  
)





UPLOAD_DIR = "uploads"
FRAMES_DIR = "static/frames"
STAT_DIR = "static"
VELOCITY_DATA_DIR = "velocity_data"
RECTANGLE_DIR = 'rectangle'

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(VELOCITY_DATA_DIR, exist_ok=True)
os.makedirs(RECTANGLE_DIR, exist_ok=True)


templates = Jinja2Templates(directory="templates")





@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_video(
    request: Request, 
    file: UploadFile = File(...),   
):
    

    session = request.session
    
    # Saving the uploaded video
    video_path = os.path.join(UPLOAD_DIR, file.filename)
    sanitized_filename = re.sub(r'[^\w\-_\. ]', '_', file.filename)
    video_path = os.path.join(UPLOAD_DIR, sanitized_filename)
    request.session['uploaded_video_path'] = video_path


    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    return RedirectResponse(url="/preview_video/", status_code=303)
    

@app.get("/preview_video/", response_class=HTMLResponse)
async def preview_video(request: Request):
    """
    Displaying video and allowing users to select start and end times.
    """
    session = request.session
    video_path = session.get("uploaded_video_path")

    if not video_path:
        return templates.TemplateResponse("index.html", {"request": request, "error": "No video uploaded."})

    video_url = f"/uploads/{Path(video_path).name}"
    #video_url = f"uploads/{os.path.basename(video_path)}"
    return templates.TemplateResponse("preview.html", {"request": request, "video_url": video_url})





@app.post("/process_video/")
async def process_video(request: Request, start_time: float = Form(...), end_time: float = Form(...)):
    """
    Processing video from start_time to end_time and extract frames.
    """
    session = request.session
    video_path = session.get("uploaded_video_path")

    if not video_path:
        return {"error": "No video uploaded."}

    try:
        frame_count = 5  
        first_frame_path, fps = extract_frames_by_time(video_path, FRAMES_DIR, start_time, end_time, frame_count=frame_count)

        session["fps"] = fps

        # Using the first frame for redirection
        first_frame_url = f"/static/frames/{os.path.basename(first_frame_path[0])}"
        redirect_url = f"/frame/?frame_url={first_frame_url}"

      

        return RedirectResponse(url=redirect_url, status_code=303)

    except Exception as e:
        return {"error": str(e)}





@app.get("/frame", response_class=HTMLResponse)
async def frame(request: Request, frame_url: str):
    print(f"Frame URL received: {frame_url}") 
    try:
       
        return templates.TemplateResponse("frame.html", {"request": request, "frame_url": frame_url})
    except Exception as e:
        print(f"Error displaying frame: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "error": "Error displaying frame"})


region_coordinates = {"x_start": 0, "y_start": 0, "x_end": 0, "y_end": 0}
is_region_selected = False


def select_region(event, x, y, flags, param):
    global region_coordinates, is_region_selected
    if event == cv2.EVENT_LBUTTONDOWN:  # Start of selection
        region_coordinates["x_start"], region_coordinates["y_start"] = x, y
    elif event == cv2.EVENT_LBUTTONUP:  # End of selection
        region_coordinates["x_end"], region_coordinates["y_end"] = x, y
        is_region_selected = True










@app.post("/select_region")
async def select_region_endpoint(request: Request, frame_file: str = Form(...), 
                                  x_start: int = Form(...), y_start: int = Form(...), 
                                  x_end: int = Form(...), y_end: int = Form(...),
                                  num_segments: int = Form(...)):
    

    
    session = request.session
    print("resssssssssssssss")
    
    


    frame_path = os.path.join(FRAMES_DIR, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Error: Failed to load frame: {frame_file}")
        return {"error": f"Failed to load frame: {frame_file}"}

    # Calculating the width of each vertical segment
    segment_width = (x_end - x_start) // num_segments

    rect_image_filename = f"rect_{os.path.basename(frame_file)}"
    rect_image_path = os.path.join(RECTANGLE_DIR, rect_image_filename)

    print(f"Saving segmented frame to: {rect_image_path}")  

    try:
        # Drawing the vertical lines to segment the selected region
        for i in range(1, num_segments):
            # Calculating the X coordinate for the vertical line
            line_x = x_start + i * segment_width
            # Drawing the vertical line on the frame
            cv2.line(frame, (line_x, y_start), (line_x, y_end), (0, 255, 0), 2)

        # Drawing the outside box (rectangle)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

        
        success = cv2.imwrite(rect_image_path, frame)
        if not success:
            print(f"Error: Failed to save the segmented frame to {rect_image_path}")
            return {"error": "Failed to save the segmented frame."}
        
        # Calculate velocity based on the selected region
        velocity_data_path = os.path.join(VELOCITY_DATA_DIR, "velocity_data.csv")
        print("velocity", velocity_data_path)

# Defining the path to the RAFT model file
        model_path = os.path.join(STAT_DIR, "raft-sintel.pth")

        calculate_velocity(
            frame_folder=FRAMES_DIR,
            output_csv=velocity_data_path,
            model_path=model_path,
            x_range=(x_start, x_end),
            y_range=(y_start, y_end),
            num_segments= num_segments
            
        )
        

    except Exception as e:
        print(f"Exception: {str(e)}")  
        return {"error": f"Exception occurred while processing: {str(e)}"}

    
    segmented_frame_url = f"/rectangle/{rect_image_filename}"



    session['segmented_frame_url'] = segmented_frame_url
    session['x_start'] = x_start
    session['y_start'] = y_start
    session['x_end'] = x_end
    session['y_end'] = y_end
    session['num_segments'] = num_segments

    






    


    
    redirect_url = f"/show_image?frame_url=/static/frames/{frame_file}&segmented_frame_url=/rectangle/{rect_image_filename}&x_start={x_start}&y_start={y_start}&x_end={x_end}&y_end={y_end}&num_segments={num_segments}"    
    print(f"Redirecting to: {redirect_url}")  
    return RedirectResponse(url=redirect_url, status_code=302)


    











@app.get("/show_image")
async def show_image(request: Request, frame_url: str = None, segmented_frame_url: str = None, 
                     x_start: int = 0, y_start: int = 0, x_end: int = 0, y_end: int = 0, real_world_distance: float = 0.0):
    print(f"Received frame_url: {frame_url}, segmented_frame_url: {segmented_frame_url}")

    


    return templates.TemplateResponse("show_image.html", {"request": request, 
                                                           "frame_url": frame_url, 
                                                           "segmented_frame_url": segmented_frame_url, 
                                                           "x_start": x_start, "y_start": y_start, 
                                                           "x_end": x_end, "y_end": y_end})







@app.post("/calculate_scaling_factor")
async def calculate_scaling_factor(
    request: Request,
    real_distance: float = Form(...),
    xmin: int = Form(...), ymin: int = Form(...),
    xmax: int = Form(...), ymax: int = Form(...),
    
):
    """
    Process both scaling factor calculation and depth data upload in a single endpoint.
    """
    try:
        #Calculating the scaling factor
        print(f"Received data - real_distance: {real_distance}, xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
        scaling_factor = calculate_scale(xmin, ymin, xmax, ymax, real_distance)
        print("Scaling_factor:", scaling_factor)

        
        
        request.session["scaling_factor"] = scaling_factor

        

        
        return RedirectResponse("/submit-arrowed-image", status_code=302)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")



@app.get("/submit-arrowed-image")
async def submit_arrowed_image(request: Request):
    
    session = request.session

    # Extract session variables
    segmented_frame_url = session.get('segmented_frame_url')
    x_start = session.get('x_start')
    y_start = session.get('y_start')
    x_end = session.get('x_end')
    y_end = session.get('y_end')
    num_segments = session.get('num_segments')
    scaling_factor = session.get('scaling_factor')
    fps = session.get('fps')

    # Checking for missing session data
    if not all([segmented_frame_url, x_start, y_start, x_end, y_end, num_segments, scaling_factor, fps]):
        return {"error": "Missing required session data."}

 

    # Loading velocity data
    velocity_csv = os.path.join(VELOCITY_DATA_DIR, "velocity_data.csv")
    if not os.path.exists(velocity_csv):
        return {"error": f"Velocity CSV file not found at {velocity_data.csv}"}

    try:
    # Reading the velocity data file
        velocity_data = pd.read_csv(velocity_csv)

    
        if 'segment' not in velocity_data.columns or 'velocity_x' not in velocity_data.columns or 'velocity_y' not in velocity_data.columns:
            return {"error": "Velocity CSV does not have the required columns: 'segment', 'velocity_x', and 'velocity_y'."}

   
    
    # Calculating average velocities for each segment
        segment_avg_velocity_x = velocity_data.groupby('segment')['velocity_x'].mean().to_dict()
        segment_avg_velocity_y = velocity_data.groupby('segment')['velocity_y'].mean().to_dict()

        


        print("Segment_avg_y:", segment_avg_velocity_y)


        total_segment_velocity_y = sum(segment_avg_velocity_y.values())

        print("Total Segment Velocity (Y):", total_segment_velocity_y)

    
        segment_avg_velocity = {
            segment: (np.sqrt(segment_avg_velocity_x[segment]**2 + segment_avg_velocity_y[segment]**2))*(scaling_factor*fps)
            for segment in segment_avg_velocity_x

                
    }
        
        
        print("Segment average velocity:", segment_avg_velocity)
        

        


        converted_velocity = {}


        # taking data from right
        for original_segment, new_segment in zip(range(num_segments, 0, -1), range(1, num_segments + 1)):
            converted_velocity[new_segment] = segment_avg_velocity[original_segment]

       
        print("Converted Velocity:", converted_velocity)
            


    except Exception as e:
        return {"error": f"Error reading or processing the velocity CSV: {str(e)}"}

    

    # Drawing arrows and save the updated image
    segmented_frame_path = os.path.join(RECTANGLE_DIR, os.path.basename(segmented_frame_url))
    frame = cv2.imread(segmented_frame_path)
    if frame is None:
        return {"error": f"Failed to load the segmented frame at {segmented_frame_path}"}

    try:
        # Drawing velocity arrows based on segments
        frame_with_arrows, _ = draw_velocity_arrows_based_on_segments(
            frame, converted_velocity, total_segment_velocity_y, x_start, y_start, x_end, y_end, num_segments
        )

        # Saving the arrowed image
        arrowed_image_filename = "rect_arrow.jpg"
        output_image_path = os.path.join(RECTANGLE_DIR, arrowed_image_filename)
        save_arrowed_image(frame_with_arrows, output_image_path)
    except Exception as e:
        return {"error": f"Error drawing arrows or saving the image: {str(e)}"}

    delete_all_files_in_directory(FRAMES_DIR)
    print("All files in the frames directory have been deleted.")

    
    session['converted_velocity'] = converted_velocity
    

   
    image_url = f"/rectangle/{arrowed_image_filename}"

   
    return RedirectResponse(url=f"/display-arrowed-image/?image_url={image_url}", status_code=302)







@app.get("/display-arrowed-image", response_class=HTMLResponse)
async def display_arrowed_image(request: Request, image_url: str):
    
    session = request.session
    converted_velocity = session.get('converted_velocity', {})
    
    return templates.TemplateResponse("average_velocity.html", {
        "request": request,
        "image_url": image_url,
        "segment_avg_velocity": converted_velocity,
      

    })



