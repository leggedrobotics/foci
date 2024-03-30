import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
import os

from .bsplines import basis_function_mat, basis_function, spline_eval
from .gaussian_robot import normal_pdf_cas, robot_obstacle_convolution, curve_robot_obstacle_convolution

def create_solver(num_control_points, n_gaussians, dim_control_points=3, num_samples=100)):
    """ Return casadi solver object and upper and lower bounds for the optimization problem
    @args:
        num_control_points: int
        n_gaussians: int
        dim_control_points: int
    """

    # define optimization parameters
    start_pos = cas.SX.sym("start_pos", dim_control_points, 1)
    end_pos = cas.SX.sym("end_pos", dim_control_points, 1)
    obstacle_means = cas.SX.sym("obstacle_means", dim_control_points, n_gaussians)
    obstacle_covs = cas.SX.sym("obstacle_covs", dim_control_points, dim_control_points, n_gaussians)
    robot_cov = cas.SX.sym("robot_cov", dim_control_points, dim_control_points,1)
    params = cas.vertcat(start_pos, end_pos, obstacle_means, obstacle_covs, robot_cov)

    # define optimization variables
    control_points = cas.SX.sym("control_points", num_control_points, dim_control_points)
    dec_vars = cas.vertcat(cas.vec(control_points))
    
    # define helpful mappings
    curve = spline_eval(control_points, num_samples)
    dcurve = spline_eval(control_points, num_samples, derivate =1)
    ddcurve = spline_eval(control_points, num_samples, derivate =2)
    
 
    # define optimization constraints
    lbg = []
    ubg = []
    cons = cas.SX([])
    
    cons = cas.vertcat(cons, (curve[0,0] - start_point[0]) ** 2 + (curve[0,1] - start_point[1]) ** 2 )
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.1]))
    
    cons = cas.vertcat(cons, (curve[-1,0] - end_point[0]) ** 2 + (curve[-1,1] - end_point[1]) ** 2 )
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.1]))


    # define optimization objective
    length_cost = 0
    for i in range(n_samples-1):
        length_cost = length_cost + (curve[i,0] - curve[i+1,0]) ** 2 + (curve[i,1] - curve[i+1,1]) ** 2

    accel_cost = cas.sum1(cas.sum2(ddcurve**2))

    obstacle_cost = curve_robot_obstacle_convolution(curve, robot_cov, obstacle_positions, obstacle_covs)
    cost = length_cost +  1000000 * obstacle_cost + 0.1 *accel_cost

    # define optimization solver
    nlp = {"x": dec_vars, "f": cost, "g": cons, "p": params}
    ipop_options = {"ipopt.print_level": 3, "ipopt.max_iter": 100, "ipopt.tol": 1e-1, "print_time": 0, "ipopt.acceptable_tol": 1e-1, "ipopt.acceptable_obj_change_tol": 1e-1, "ipopt.hessian_approximation": "limited-memory", "ipopt.mu_strategy": "adaptive"}

    solver = cas.nlpsol("solver", "ipopt", nlp, ipop_options)

    return solver, lbg, ub
 
    




