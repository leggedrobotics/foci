import matplotlib
import numpy as np
import os
import logging

logging.basicConfig(level =logging.INFO)

def basis_function(t, derivate = 0):
    assert derivate in [0,1,2,3], "derivate must be 0, 1, 2, or 3"

    if derivate == 0:
      t_ = np.array([1, t, t**2, t**3])
    if derivate == 1:
       t_ = np.array([0, 1, 2*t, 3*t**2])
    if derivate == 2:
       t_ = np.array(([0,0, 2, 6* t]))
    if derivate == 3:
       t_ = np.array(([0,0,0, 6])) 
    
    feature_matrix = 1/6* np.array([[1, 4, 1, 0],
                               [-3,0 ,3,0],
                                 [3,-6,3,0],
                                 [-1,3,-3,1]])
    return t_ @ feature_matrix

def basis_function_mat(ts, n_control_points, n_knots, derivate = 0):
    mat = np.zeros((ts.shape[0], n_control_points))
    for i, t in enumerate(ts):
        offset = max(min(int(np.floor(t)), n_knots -1),0)
        u = t - offset
        mat[i][offset: offset + 4] = basis_function(u, derivate = derivate)
    
    return mat

def spline_eval(control_points, num_samples, derivate = 0):
    n_knots = control_points.shape[0] - 4

    ts = np.linspace(0, n_knots, num_samples)
    basis = basis_function_mat(ts, control_points.shape[0], n_knots, derivate = derivate)
    curve = basis @ control_points

    return curve


def spline_eval_at_s(control_points, s, derivate = 0):
    n_knots = control_points.shape[0] - 4

    ts = np.array([s])
    basis = basis_function_mat(ts, control_points.shape[0], n_knots, derivate = derivate)
    curve = basis @ control_points
    
    point = curve[0]

    return point

