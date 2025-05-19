import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

import pandas as pd

logging.basicConfig(level =logging.INFO)

from foci.splines.bsplines import  spline_eval, get_minvo_hulls
from foci.convolution.gaussian_robot_warp import ConvolutionFunctorWarp

def create_solver(num_control_points,
                obstacle_means, 
                covs_det, 
                covs_inv, 
                kinematics,  
                dim_control_points=3,
                dim_rotation = 1,
                num_samples=30, 
                num_body_parts = 1, 
                x_range = None, 
                y_range = None, 
                z_range = None,
                vmax = 1, 
                amax = 1):
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


    start_position = start_pos[:3]
    end_position = end_pos[:3]
    T = cas.norm_2(end_pos - start_pos)/vmax # estimate of time to reach goal

    S = num_control_points - 4  # number of knots, spline is parametrized on interval [0, S]
    m_t_to_s = S/T # scaling factor to convert time to spline parameter


    print("m_t_to_s", m_t_to_s.shape)

    # convert m_t_to_s to casadi type SX


    # Weighing factor for cost
    w_0 = 0.1 # jerk cost
    w_1 = 40 # obstacle cost
    w_2 = 1 # goal cost
    
    # define helpful mappings
    curve = spline_eval(control_points, num_samples)
    dcurve = m_t_to_s * spline_eval(control_points, num_samples, derivate = 1)
    ddcurve = (m_t_to_s **2)  * spline_eval(control_points, num_samples, derivate =2) 
    dddcurve = (m_t_to_s **3) * spline_eval(control_points, num_samples, derivate =3) 

    pos_hulls = get_minvo_hulls(control_points, derivative = 0)
    vel_hulls = [m_t_to_s * hull for hull in get_minvo_hulls(control_points, derivative = 1)]
    acc_hulls = [(m_t_to_s **2) * hull for hull in get_minvo_hulls(control_points, derivative = 2)]
    jerk_hulls = [(m_t_to_s **3) * hull for hull in get_minvo_hulls(control_points, derivative = 3)]


    kinematics_functor = kinematics.map(num_samples, "openmp") 

    # define optimization constraints
    lbg = []
    ubg = []
    cons = SYM_TYPE([])


    # start start pose contraint 
    cons = cas.vertcat(cons, (curve[0,0] - start_pos[0]) ** 2 + (curve[0,1] - start_pos[1]) ** 2 + (curve[0,2] - start_pos[2]) ** 2 + (curve[0,3] - start_pos[3]) ** 2)
    lbg = np.concatenate((lbg, [0]))
    ubg = np.concatenate((ubg, [0.05]))

    # Position constraints =================================
    # if x_range is not None:
    #     for i in range(curve.shape[0]):
    #         cons = cas.vertcat(cons, curve[i,0])
    #         lbg = np.concatenate((lbg, [x_range[0]]))
    #         ubg = np.concatenate((ubg, [x_range[1]]))
    
    # if y_range is not None:
    #     for i in range(curve.shape[0]):
    #         cons = cas.vertcat(cons, curve[i,1])
    #         lbg = np.concatenate((lbg, [y_range[0]]))
    #         ubg = np.concatenate((ubg, [y_range[1]]))

    # if z_range is not None:
    #     for i in range(curve.shape[0]):
    #         cons = cas.vertcat(cons, curve[i,2])
    #         lbg = np.concatenate((lbg, [z_range[0]]))
    #         ubg = np.concatenate((ubg, [z_range[1]]))
    
    if x_range is not None:
        for hull in pos_hulls:
            for i in range(hull.shape[0]):
                cons = cas.vertcat(cons, hull[i,0])
                lbg = np.concatenate((lbg, [x_range[0]]))
                ubg = np.concatenate((ubg, [x_range[1]]))

    if y_range is not None:
        for hull in pos_hulls:
            for i in range(hull.shape[0]):
                cons = cas.vertcat(cons, hull[i,1])
                lbg = np.concatenate((lbg, [y_range[0]]))
                ubg = np.concatenate((ubg, [y_range[1]]))
    
    if z_range is not None:
        for hull in pos_hulls:
            for i in range(hull.shape[0]):
                cons = cas.vertcat(cons, hull[i,2])
                lbg = np.concatenate((lbg, [z_range[0]]))
                ubg = np.concatenate((ubg, [z_range[1]]))

            
    # velocity constraints =================================
    # for i in range(curve.shape[0]):
    #     cons = cas.vertcat(cons, dcurve[i,0] ** 2 + dcurve[i,1] ** 2 + dcurve[i,2] ** 2)
    #     lbg = np.concatenate((lbg, [0]))
    #     if i == curve.shape[0] - 1:
    #         ubg = np.concatenate((ubg, [0]))
    #     else:
    #         ubg = np.concatenate((ubg, [vmax ** 2]))

    for hull in vel_hulls:
        for i in range(hull.shape[0]):
            cons = cas.vertcat(cons, hull[i,0] ** 2 + hull[i,1] ** 2 + hull[i,2] ** 2)
            lbg = np.concatenate((lbg, [0]))
            ubg = np.concatenate((ubg, [vmax ** 2]))

    
    
    # acceleration constraints =============================
    # for i in range(curve.shape[0]):
    #     cons = cas.vertcat(cons, ddcurve[i,0] ** 2 + ddcurve[i,1] ** 2 + ddcurve[i,2] ** 2)
    #     lbg = np.concatenate((lbg, [0]))
    #     if i == curve.shape[0] - 1:
    #         ubg = np.concatenate((ubg, [0]))
    #     else:
    #         ubg = np.concatenate((ubg, [amax ** 2]))

    for hull in acc_hulls:
        for i in range(hull.shape[0]):
            cons = cas.vertcat(cons, hull[i,0] ** 2 + hull[i,1] ** 2 + hull[i,2] ** 2)
            lbg = np.concatenate((lbg, [0]))
            ubg = np.concatenate((ubg, [amax ** 2]))


    collision_points = kinematics_functor(curve.T).T

    convolution_functor = ConvolutionFunctorWarp("conv",dim_control_points -1,num_body_parts * num_samples, obstacle_means, covs_det, covs_inv)
    obstacle_cost = convolution_functor(collision_points)


    jerk_cost = cas.sum1(cas.sum2(dddcurve **2))

    goal_cost = (curve[-1,0] - end_pos[0]) ** 2 + (curve[-1,1] - end_pos[1]) ** 2 + (curve[-1,2] - end_pos[2]) ** 2 

    cost =  w_0 *jerk_cost + w_1 * obstacle_cost + w_2 * goal_cost

    # define optimization solver
    nlp = {"x": dec_vars, "f": cost, "p": params, "g": cons}
    ipopt_options = {"ipopt.print_level": 5,
                    "ipopt.max_iter":200, 
                    "ipopt.tol": 1e-1, 
                    "print_time": 0, 
                    "ipopt.acceptable_tol": 1e-3, 
                    "ipopt.acceptable_obj_change_tol": 1e-1,
                    "ipopt.constr_viol_tol": 1e-3,
                    "ipopt.acceptable_iter": 3,
                    "ipopt.linear_solver": "ma27",
                    "ipopt.hessian_approximation": "limited-memory",
                    }

    solver = cas.nlpsol("solver", "ipopt", nlp, ipopt_options) 

    return solver, lbg,ubg , convolution_functor



