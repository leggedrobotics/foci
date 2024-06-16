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

def create_solver(num_control_points, obstacle_means, covs_det, covs_inv, kinematics,  dim_control_points=3, dim_rotation = 1,num_samples=30, num_body_parts = 1, x_range = None, y_range = None, z_range = None):
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

    dim_control_points = dim_control_points + dim_rotation

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
    dcurve = spline_eval(control_points, num_samples, derivate = 1)
    ddcurve = spline_eval(control_points, num_samples, derivate =2)


    kinematics_functor = kinematics.map(num_samples, "openmp") 

    # define optimization constraints
    lbg = []
    ubg = []
    cons = SYM_TYPE([])
    
    cons = cas.vertcat(cons, (curve[0,0] - start_pos[0]) ** 2 + (curve[0,1] - start_pos[1]) ** 2 + (curve[0,2] - start_pos[2]) ** 2 + (curve[0,3] - start_pos[3]) ** 2)
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.05]))

    # # constraint start velocity to zero
    # cons = cas.vertcat(cons, (dcurve[0,0] **2 + dcurve[0,1] **2 + dcurve[0,2] **2 + dcurve[0,3] **2))
    # lbg = np.concatenate((lbg, [0]))
    # ubg = np.concatenate((ubg, [0.05]))
    
    
    cons = cas.vertcat(cons, (curve[-1,0] - end_pos[0]) ** 2 + (curve[-1,1] - end_pos[1]) ** 2 + (curve[-1,2] - end_pos[2]) ** 2 + (curve[-1,3] - end_pos[3]) ** 2)
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.05]))

    # # constraint end velocity to zero
    # cons = cas.vertcat(cons, (dcurve[-1,0] **2 + dcurve[-1,1] **2 + dcurve[-1,2] **2 + dcurve[-1,3] **2))
    # lbg = np.concatenate((lbg, [0]))
    # ubg = np.concatenate((ubg, [0.05]))

    if x_range is not None:
        for i in range(curve.shape[0]):
            cons = cas.vertcat(cons, curve[i,0])
            lbg = np.concatenate((lbg, [x_range[0]]))
            ubg = np.concatenate((ubg, [x_range[1]]))
    
    if y_range is not None:
        for i in range(curve.shape[0]):
            cons = cas.vertcat(cons, curve[i,1])
            lbg = np.concatenate((lbg, [y_range[0]]))
            ubg = np.concatenate((ubg, [y_range[1]]))
                                 


    if z_range is not None:
        for i in range(curve.shape[0]):
            cons = cas.vertcat(cons, curve[i,2])
            lbg = np.concatenate((lbg, [z_range[0]]))
            ubg = np.concatenate((ubg, [z_range[1]]))

    # define optimization objective
    accel_cost = cas.sum1(cas.sum2(ddcurve[:,:2]**2))  + 0.1 * cas.sum1(cas.sum2(ddcurve[:,3] ** 2))# TODO: handle rotation seperately

    length_cost = 0
    for i in range(num_samples - 1):
        length_cost += cas.sqrt((curve[i+1,0] - curve[i,0])**2 + (curve[i+1,1] - curve[i,1])**2 + (curve[i+1,2] - curve[i,2])**2)

    collision_points = kinematics_functor(curve.T).T

    convolution_functor = ConvolutionFunctorWarp("conv",dim_control_points -1,num_samples*num_body_parts, obstacle_means, covs_det, covs_inv)
    obstacle_cost = convolution_functor(collision_points)

    cost = accel_cost  + 0.01* length_cost 
    cons = cas.vertcat(cons, obstacle_cost)
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.1]))
    
    # define optimization solver
    nlp = {"x": dec_vars, "f": cost, "p": params, "g": cons}
    ipopt_options = {"ipopt.print_level": 5,
                    "ipopt.max_iter":100, 
                    "ipopt.tol": 1e-1, 
                    "print_time": 0, 
                    "ipopt.acceptable_tol": 1e-1, 
                    "ipopt.acceptable_obj_change_tol": 1e-1,
                    "ipopt.constr_viol_tol": 1e-1,
                    "ipopt.acceptable_iter": 1,
                    "ipopt.linear_solver": "ma27",
                    "ipopt.hessian_approximation": "limited-memory",
                    }

    solver = cas.nlpsol("solver", "ipopt", nlp, ipopt_options) 

    return solver, lbg,ubg , convolution_functor



