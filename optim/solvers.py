import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

import pandas as pd

logging.basicConfig(level =logging.INFO)

from gsplat_traj_optim.splines.bsplines import  spline_eval
from gsplat_traj_optim.convolution.gaussian_robot import create_curve_robot_obstacle_convolution_functor

def create_solver(num_control_points, n_gaussians, dim_control_points=3, num_samples=30, sym_type="SX"):
    """ Return casadi solver object and upper and lower bounds for the optimization problem
    @args:
        num_control_points: int
        n_gaussians: int
        dim_control_points: int
        num_samples: int
        sym_type: str, "SX" or "MX"
    @returns:
        solver: casadi solver object
        lbg: np.array
        ubg: np.array
    """

    SYM_TYPE = cas.MX  

    # define optimization parameters
    start_pos = SYM_TYPE.sym("start_pos", dim_control_points, 1)
    end_pos = SYM_TYPE.sym("end_pos", dim_control_points, 1)
    obstacle_means = SYM_TYPE.sym("obstacle_means", dim_control_points, n_gaussians)
    obstacle_covs = SYM_TYPE.sym("obstacle_covs", dim_control_points * dim_control_points, n_gaussians)
    robot_cov = SYM_TYPE.sym("robot_cov", dim_control_points* dim_control_points,1)
    params = cas.vertcat(cas.vec(start_pos),
                    cas.vec(end_pos),
                    cas.vec(obstacle_means),
                    cas.vec(obstacle_covs),
                    cas.vec(robot_cov)
    )

    # define optimization variables
    control_points = SYM_TYPE.sym("control_points", num_control_points, dim_control_points)
    dec_vars = cas.vertcat(cas.vec(control_points))
    
    # define helpful mappings
    curve = spline_eval(control_points, num_samples)
    dcurve = spline_eval(control_points, num_samples, derivate =1)
    ddcurve = spline_eval(control_points, num_samples, derivate =2)
    
 
    # define optimization constraints
    lbg = []
    ubg = []
    cons = SYM_TYPE([])
    
    cons = cas.vertcat(cons, (curve[0,0] - start_pos[0]) ** 2 + (curve[0,1] - start_pos[1]) ** 2 + (curve[0,2] - start_pos[2]) ** 2)
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.1]))
    
    cons = cas.vertcat(cons, (curve[-1,0] - end_pos[0]) ** 2 + (curve[-1,1] - end_pos[1]) ** 2 + (curve[-1,2] - end_pos[2]) ** 2)
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.1]))


    # define optimization objective
    length_cost = 0
    for i in range(num_samples-1):
        length_cost = length_cost + (curve[i,0] - curve[i+1,0]) ** 2 + (curve[i,1] - curve[i+1,1]) ** 2

    accel_cost = cas.sum1(cas.sum2(ddcurve**2))

    convolution_functor = create_curve_robot_obstacle_convolution_functor(num_samples, n_gaussians, dim_control_points)
    obstacle_cost = convolution_functor(curve, robot_cov, obstacle_means, obstacle_covs)

    cost = length_cost +  1000000 * obstacle_cost + 0.1 *accel_cost

    # define optimization solver
    nlp = {"x": dec_vars, "f": cost, "g": cons, "p": params}
    ipop_options = {"ipopt.print_level": 3, "ipopt.max_iter": 100, "ipopt.tol": 1e-1, "print_time": 0, "ipopt.acceptable_tol": 1e-1, "ipopt.acceptable_obj_change_tol": 1e-1, "ipopt.hessian_approximation": "limited-memory", "ipopt.mu_strategy": "adaptive"}

    solver = cas.nlpsol("solver", "ipopt", nlp, ipop_options)

    return solver, lbg, ubg



