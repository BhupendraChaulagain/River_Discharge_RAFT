# import math


def calculate_scale(xmin, ymin, xmax, ymax, real_distance):


    pixel_distance = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5


    scaling_factor = (real_distance/ (pixel_distance)) if pixel_distance !=0 else 0

    print(f"real_distance: {real_distance}, Pixel Distance: {pixel_distance}, Scaling Factor: {scaling_factor}")
    return scaling_factor