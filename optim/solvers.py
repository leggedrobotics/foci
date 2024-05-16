import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

import pandas as pd

logging.basicConfig(level =logging.INFO)

from gsplat_traj_optim.splines.bsplines import  spline_eval
from gsplat_traj_optim.convolution.gaussian_robot import create_curve_robot_obstacle_convolution_functor
from gsplat_traj_optim.convolution.gaussian_robot_warp import ConvolutionFunctorWarp

def create_solver(num_control_points, obstacle_means, covs_det, covs_inv,  dim_control_points=3, num_samples=30):
    """ Return casadi solver object and upper and lower bounds for the optimization problem
    @args:
        num_control_points: int
        n_gaussians: int
        dim_control_points: int
        num_samples: int
        expand: bool
    @returns:
        solver: casadi solver object
        lbg: np.array
        ubg: np.array
    """

    SYM_TYPE = cas.MX  

    # define optimization parameters
    start_pos = SYM_TYPE.sym("start_pos", dim_control_points, 1)
    end_pos = SYM_TYPE.sym("end_pos", dim_control_points, 1)
    params = cas.vertcat(cas.vec(start_pos),
                    cas.vec(end_pos),
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
    ubg = np.concatenate((ubg, [0.05]))
    
    cons = cas.vertcat(cons, (curve[-1,0] - end_pos[0]) ** 2 + (curve[-1,1] - end_pos[1]) ** 2 + (curve[-1,2] - end_pos[2]) ** 2)
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.05]))


    for i in range(num_samples):
        cons = cas.vertcat(cons, curve[i,1])
        lbg = np.concatenate((lbg, [5]))
        ubg = np.concatenate((ubg, [7]))
        
        cons = cas.vertcat(cons, curve[i,2])
        lbg = np.concatenate((lbg, [-10]))
        ubg = np.concatenate((ubg, [3]))



    # define optimization objective
    length_cost = 0
    for i in range(num_samples-1):
        length_cost = length_cost + (curve[i,0] - curve[i+1,0]) ** 2 + (curve[i,1] - curve[i+1,1]) ** 2

    accel_cost = cas.sum1(cas.sum2(ddcurve**2))
    vel_cost = cas.sum1(cas.sum2(dcurve**2))


    convolution_functor = ConvolutionFunctorWarp("conv",dim_control_points,num_samples, obstacle_means, covs_det, covs_inv )
    obstacle_cost = convolution_functor(curve)

    cost =   accel_cost


    cons = cas.vertcat(cons, obstacle_cost)
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [2]))



    # define optimization solver
    nlp = {"x": dec_vars, "f": cost, "p": params, "g": cons}
    ipopt_options = {"ipopt.print_level": 3,
                    "ipopt.max_iter":100, 
                    "ipopt.tol": 1e-1, 
                    "print_time": 0, 
                    "ipopt.acceptable_tol": 1e-1, 
                    "ipopt.acceptable_obj_change_tol": 1e-2,
                    "ipopt.constr_viol_tol": 1e-2,
                    "ipopt.acceptable_iter": 1,
                    "ipopt.linear_solver": "ma27",
                    "ipopt.hessian_approximation": "limited-memory",
                    }

    solver = cas.nlpsol("solver", "ipopt", nlp, ipopt_options) 

    return solver, lbg,ubg , convolution_functor



