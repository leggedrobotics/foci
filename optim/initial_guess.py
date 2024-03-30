import numpy as np

def linear_interpolation(start_point, end_point, n_control_points):
    """Return the control points of a linear interpolation between start and end point
    @args:
        start_point: np.array
        end_point: np.array
        n_control_points: int
    """
    control_points = np.zeros((n_control_points, start_point.shape[0]))
    for i in range(n_control_points):
        control_points[i] = start_point + i/(n_control_points - 1) * (end_point - start_point)
    return control_points
