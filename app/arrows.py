import cv2

def draw_velocity_arrows_based_on_segments(frame, segment_avg_velocity, total_y_velocity, x_start, y_start, x_end, y_end, num_segments):
    # Height and width calculation
    segment_width = (x_end - x_start) // num_segments
    segment_height = y_end - y_start

    area = segment_width * segment_height

    max_velocity = max(segment_avg_velocity.values())
    #finding maximum velocity
    scaling_factor = segment_height / max_velocity if max_velocity > 0 else 1

    for segment, velocity in segment_avg_velocity.items():
        # horizontal space finding
        segment_x_end = x_end - (segment - 1) * segment_width  
        segment_x_center = segment_x_end - segment_width // 2  # Center of the segment

        #arrow making on the basis of velocity in y direction

        if total_y_velocity >= 0:

            start_point = (segment_x_center, y_start)

        # Ending arrow
            end_point = (segment_x_center, y_start + int(velocity * scaling_factor))

        else:
            start_point = (segment_x_center, y_end)

            end_point = (segment_x_center, y_end - int(velocity * scaling_factor))

        # Drawing arrow
        cv2.arrowedLine(
            frame,
            start_point,
            end_point,
            color=(255, 0, 0),  # Blue color in BGR
            thickness=5,
            tipLength=0.2
        )
        
    return frame, area
