import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import casadi as cas

from gsplat_traj_optim.splines.bsplines import spline_eval, spline_eval_at_s
from gsplat_traj_optim.optim.initial_guess import linear_interpolation, astar_path_spline_fit 
from gsplat_traj_optim.convolution.gaussian_robot_warp import ConvolutionFunctorWarp
from gsplat_traj_optim.visualisation.vis_utils import EnvAndPathVis
from gsplat_traj_optim.optim.solvers import create_solver

from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize

import rospy



class Planner():
    def __init__(self, obstacle_positions, obstacle_covs, robot_cov, num_control_points, num_samples):  

        self.num_control_points = num_control_points
        self.num_samples = num_samples
        self.obstacle_positions = obstacle_positions
        self.obstacle_covs = obstacle_covs
        self.robot_cov = robot_cov
    

        # casadi function to convert form pose (x y z theta) to 3 positions
        pose = cas.MX.sym("pose", 4)
        theta = pose[3]
        middle = pose[:3]
        scale = 0.5
        left = middle - cas.vertcat(cas.cos(theta)* scale, cas.sin(theta) * scale, 0)
        right = middle + cas.vertcat(cas.cos(theta) *scale, cas.sin(theta) * scale, 0)

        # Create casadi function
        self.kinematics = cas.Function("kinematics", [pose], [cas.horzcat(left, middle, right)])

        
        N = obstacle_positions.shape[0]
        num_samples = 30

        covs_sum = obstacle_covs + robot_cov 
        covs_det = np.zeros(len(covs_sum))
        covs_inv = np.zeros_like(covs_sum)
        for i in range(len(covs_sum)):
            covs_det[i] = np.linalg.det(covs_sum[i])
            covs_inv[i] = np.linalg.inv(covs_sum[i])


        self.solver, self.lbg, self.ubg, self.convolution_functor = create_solver(num_control_points, obstacle_positions, covs_det, covs_inv, self.kinematics, dim_control_points=3, num_samples=num_samples)




    def plan(self,start_pos, end_pos):
 
        rospy.loginfo("Computing initial guess")
        init_guess = astar_path_spline_fit( start_pos, end_pos, self.obstacle_positions, num_control_points=self.num_control_points, voxel_size=1.5) 
        rospy.loginfo("Initial guess computed")
        spline = spline_eval((init_guess.reshape(4, self.num_control_points)).T, self.num_samples)

    
        rospy.loginfo("Optimizing")
        res = self.solver(x0 = init_guess, lbg = self.lbg, ubg = self.ubg, p = np.concatenate((start_pos, end_pos)))
        self.control_points_opt = np.array(res['x']).reshape(4, self.num_control_points).T
        opt_curve = spline_eval(self.control_points_opt, self.num_samples)
        rospy.loginfo("Optimization done")

        vis = EnvAndPathVis()
        vis.add_points(self.obstacle_positions, color = [0,0,1])    
        vis.add_curve(spline[:,:3], color = [1,0,0])
        vis.add_gaussian_path(opt_curve, self.robot_cov ,self.kinematics,color = [0,1,0])
        vis.show()

        return opt_curve

    def regularize(self,max_vel):
        vel_curve = spline_eval(self.control_points_opt, self.num_samples *30, derivate = 1) 
        rospy.loginfo(vel_curve.shape)
        max_ds = np.max(np.linalg.norm(vel_curve, axis = 1))
        self.a = max_vel / max_ds
        rospy.loginfo("Regularization factor: %f", self.a)

    def evaluate_x(self,t):
        s = self.a * t
        return spline_eval_at_s(self.control_points_opt, s)


    def evaluate_dx(self,t):
        s = self.a * t
        return self.a * spline_eval_at_s(self.control_points_opt, s, derivate =1)

    def max_time(self):
        return (1/self.a) * (self.num_control_points-4)
 