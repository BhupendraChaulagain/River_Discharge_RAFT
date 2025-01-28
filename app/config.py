scaling_factor = None
fps = 20
num_segments = 5

def get_scaling_factor():
    return scaling_factor if scaling_factor is not None else 1.0, fps, num_segments