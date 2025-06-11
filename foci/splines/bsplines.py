import matplotlib
import numpy as np
import os
import logging
import casadi as cas

from scipy.spatial import ConvexHull

logging.basicConfig(level =logging.INFO)


# ==================================== FEATURE MATRICIES ====================================

MINVO_3 = np.array([
    [-3.4416309793565660335445954842726,  6.9895482693324069156659561485867, -4.4622887879670974919932291413716,                  0.91437149799125659734933338484986],
    [ 6.6792587678886103930153694818728, -11.845989952130473454872117144987,  5.2523596862506065630071816485724, -0.000000000000000055511151231257827021181583404541],
    [-6.6792587678886103930153694818728,  8.1917863515353577241739913006313, -1.5981560856554908323090558042168,                 0.085628502008743445639282754200394],
    [ 3.4416309793565660335445954842726,  -3.335344668737291184967830304231, 0.80808518737198176129510329701588, -0.000000000000000012522535092207576212786079850048],
]).T

MINVO_2 = np.array([
    [ 1.4999999992328318931811281800037, -2.3660254034601951866889635311964,   0.9330127021136816189983420599674],
    [-2.9999999984656637863622563600074,  2.9999999984656637863622563600074,                                   0],  
    [ 1.4999999992328318931811281800037, -0.6339745950054685996732928288111, 0.066987297886318325490506708774774],
]).T

MINVO_1 = np.array([
    [-1, 1],
    [1,0],
]).T

MINVO_0 = np.array([
    [1],
]).T

BSPLINE_3 = np.array([
    [-0.16666666666666666666666666666667,  0.5, -0.5, 0.16666666666666666666666666666667],
    [                                0.5, -1.0,    0, 0.66666666666666666666666666666667],
    [                               -0.5,  0.5,  0.5, 0.16666666666666666666666666666667],
    [ 0.16666666666666666666666666666667,    0,    0,                                  0]

]).T

BSPLINE_2 = np.array([
    [ 0.5, -1.0, 0.5],
    [-1.0,  1.0, 0.5],
    [ 0.5,    0,   0]
]).T 


B_SPLINE_1 = np.array([ 
    [-1, 1],
    [1, 0]
]).T

B_SPLINE_0 = np.array([
    [1]
]).T


def basis_function(t, derivate = 0):
    assert derivate in [0,1,2,3], "derivate must be 0, 1 or 2"

    if derivate == 0:
      t_ = np.array([t**3, t**2, t, 1])
    if derivate == 1:
       t_ = np.array([3 *t**2, 2*t, 1,0])
    if derivate == 2:
       t_ = np.array([6*t,2, 0, 0])
    if derivate == 3:
        t_ = np.array([6,0, 0, 0])

    return t_ @ BSPLINE_3

def basis_function_mat(ts, n_control_points, upper_bound, derivate = 0):
    mat = np.zeros((ts.shape[0], n_control_points))
    for i, t in enumerate(ts):
        offset = max(min(int(np.floor(t)), upper_bound -1),0)
        u = t - offset
        mat[i][offset: offset + 4] = basis_function(u, derivate = derivate)
    
    return mat

def spline_eval(control_points, num_samples, derivate = 0):
    upper_bound = control_points.shape[0] - 3

    ts = np.linspace(0, upper_bound, num_samples)
    basis = basis_function_mat(ts, control_points.shape[0], upper_bound, derivate = derivate)
    curve = basis @ control_points

    return curve


def spline_eval_at_s(control_points, s, derivate = 0):
    upper_bound = control_points.shape[0] - 4

    ts = np.array([s])
    basis = basis_function_mat(ts, control_points.shape[0], upper_bound, derivate = derivate)
    curve = basis @ control_points
    
    point = curve[0]

    return point




def get_minvo_hulls(control_points, derivative = 0):
    assert derivative in [0,1,2,3,], "higher order derivatives not supported"

    control_points_type = type(control_points).__name__


    V = control_points
    dV = cas.MX.zeros(control_points.shape[0]-1, control_points.shape[1]) if control_points_type == "MX" else np.zeros((control_points.shape[0]-1, control_points.shape[1]))
    ddV = cas.MX.zeros(control_points.shape[0]-2, control_points.shape[1]) if control_points_type == "MX" else np.zeros((control_points.shape[0]-2, control_points.shape[1]))
    dddV = cas.MX.zeros(control_points.shape[0]-3, control_points.shape[1]) if control_points_type == "MX" else np.zeros((control_points.shape[0]-3, control_points.shape[1]))


    if derivative >= 1:
        for i in range(control_points.shape[0]-1):
            dV[i,:] =  (control_points[i +1,:] - control_points[i,:])
    if derivative >= 2:
        for i in range(control_points.shape[0]-2):
            ddV[i,:] =  (dV[i +1,:] - dV[i,:])
    if derivative >= 3:
        for i in range(control_points.shape[0]-3):
            dddV[i,:] =  (ddV[i +1,:] - ddV[i, :])

    hulls = [] 
    if derivative == 0:
        for i in range(control_points.shape[0]-3):
            hull = np.linalg.inv(MINVO_3) @ BSPLINE_3 @ V[i:i+4,:]
            # hull = V[i:i+4,:]
            hulls.append(hull)

    elif derivative == 1:
        for i in range(control_points.shape[0]-4):
            hull = np.linalg.inv(MINVO_2) @ BSPLINE_2 @ dV[i:i+3,:]
            # hull = dV[i:i+3,:]
            hulls.append(hull)

    elif derivative == 2:
        for i in range(control_points.shape[0]-5):
            hull = np.linalg.inv(MINVO_1) @ B_SPLINE_1 @ ddV[i:i+2,:]
            # hull = ddV[i:i+2,:]
            hulls.append(hull)

    elif derivative == 3:
        for i in range(control_points.shape[0]-6):
            hull = np.linalg.inv(MINVO_0) @ B_SPLINE_0 @ dddV[i:i+1,:]
            # hull = dddV[i:i+1,:]
            hulls.append(hull)

    return hulls




if __name__ == "__main__":
    from scipy.interpolate import BSpline
    deg = 3
    knots = [ -3, -2,  -1,  0,  1, 2,   3,    4,    5,  6,  7, 8, 9, 10];
    ctrl_pts = [0,1,2,3,4,5,6,7,8,9,10];
    pos = BSpline(knots, ctrl_pts, deg)
    vel=pos.derivative(1);
    accel=pos.derivative(2);
    jerk=pos.derivative(3);
    print("Pos= ",pos(0.5))
    print("Vel= ",vel(0.5))
    print("Accel= ",accel(0.5))
    print("Jerk= ",jerk(0.5))

    # Test the spline evaluation
    control_points = np.array((ctrl_pts)).reshape(-1,1)

    # control_points[0,0] = 100
    
    p = spline_eval_at_s(control_points, 0.5, derivate = 0)
    v = spline_eval_at_s(control_points, 0.5, derivate = 1)
    a = spline_eval_at_s(control_points, 0.5, derivate = 2) 

    print("p = ", p)
    print("v = ", v)
    print("a = ", a)

    for i in range(0, 70):
        p = spline_eval_at_s(control_points, i/10, derivate = 0)
        v = spline_eval_at_s(control_points, i/10, derivate = 1)
        a = spline_eval_at_s(control_points, i/10, derivate = 2)

        assert np.allclose(p, pos(i/10))
        assert np.allclose(v, vel(i/10))
        assert np.allclose(a, accel(i/10))
    