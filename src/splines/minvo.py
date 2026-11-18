import numpy as np
import casadi as cas

from src.splines.basis import BSPLINE_3, BSPLINE_2, BSPLINE_1, BSPLINE_0
from src.splines.basis import INV_MINVO_3, INV_MINVO_2, INV_MINVO_1, INV_MINVO_0


def _forward_differences(points):
    """
    Compute first forward differences of a sequence of points.

    Supports both NumPy arrays and CasADi MX matrices.

    Parameters
    ----------
    points : np.ndarray or cas.MX
        Array of shape (n, dim).

    Returns
    -------
    np.ndarray or cas.MX
        Forward differences of shape (n - 1, dim).
    """
    points_type = type(points).__name__

    if points_type == "MX":
        result = cas.MX.zeros(points.shape[0] - 1, points.shape[1])
    else:
        result = np.zeros((points.shape[0] - 1, points.shape[1]))

    for i in range(points.shape[0] - 1):
        result[i, :] = points[i + 1, :] - points[i, :]

    return result


def minvo_hulls(control_points, derivative_order=0):
    """
    Compute MINVO control hulls associated with a cubic B-spline and its derivatives.

    Parameters
    ----------
    control_points : np.ndarray or cas.MX
        Control point array of shape (num_control_points, dim).
    derivative_order : int, default=0
        Derivative order. Supported values: 0, 1, 2, 3.

    Returns
    -------
    list
        List of hull control matrices, one per spline segment.

    Raises
    ------
    ValueError
        If `derivative_order` is not in {0, 1, 2, 3}.
    """

    if derivative_order not in {0, 1, 2, 3}:
        raise ValueError(
            f"derivative_order must be one of {{0, 1, 2, 3}}, got {derivative_order}"
        )
    first_diff = _forward_differences(control_points)
    second_diff = _forward_differences(first_diff) if derivative_order >= 2 else None
    third_diff = _forward_differences(second_diff) if derivative_order >= 3 else None

    hulls = []

    if derivative_order == 0:
        for i in range(control_points.shape[0] - 3):
            hull = INV_MINVO_3 @ BSPLINE_3 @ control_points[i : i + 4, :]
            hulls.append(hull)

    elif derivative_order == 1:
        for i in range(control_points.shape[0] - 3):
            hull = INV_MINVO_2 @ BSPLINE_2 @ first_diff[i : i + 3, :]
            hulls.append(hull)

    elif derivative_order == 2:
        for i in range(control_points.shape[0] - 3):
            hull = INV_MINVO_1 @ BSPLINE_1 @ second_diff[i : i + 2, :]
            hulls.append(hull)

    else:
        for i in range(control_points.shape[0] - 3):
            hull = INV_MINVO_0 @ BSPLINE_0 @ third_diff[i : i + 1, :]
            hulls.append(hull)

    return hulls
