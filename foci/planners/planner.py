import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import casadi as cas

from foci.splines.bsplines import spline_eval, spline_eval_at_s
from foci.optim.initial_guess import linear_interpolation, astar_path_spline_fit 
from foci.convolution.gaussian_robot_warp import ConvolutionFunctorWarp
from foci.visualisation.vis_utils import ViserVis
from foci.optim.solvers import create_solver

from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize

import rospy



class Planner():
    def __init__(self, obstacle_positions, obstacle_covs, robot_cov, num_control_points, num_samples, obstacle_colors = None):  

        self.num_control_points = num_control_points
        self.num_samples = num_samples
        self.obstacle_positions = obstacle_positions
        self.obstacle_covs = obstacle_covs
        self.robot_cov = robot_cov
        self.obstacle_colors = obstacle_colors
    

        # casadi function to convert form pose (x y z theta) to 3 positions
        pose = cas.MX.sym("pose", 4)
        theta = pose[3]
        middle = pose[:3]
        scale = 0.5
        left = middle - cas.vertcat(cas.cos(theta)* scale, cas.sin(theta) * scale, 0)
        right = middle + cas.vertcat(cas.cos(theta) *scale, cas.sin(theta) * scale, 0)

        # Create casadi function
        self.kinematics = cas.Function("kinematics", [pose], [cas.horzcat(left, middle, right)])

        

        covs_sum = obstacle_covs + robot_cov 
        covs_det = np.zeros(len(covs_sum))
        covs_inv = np.zeros_like(covs_sum)
        for i in range(len(covs_sum)):
            covs_det[i] = np.linalg.det(covs_sum[i])
            covs_inv[i] = np.linalg.inv(covs_sum[i])


        self.solver, self.lbg, self.ubg, self.convolution_functor = create_solver(num_control_points, obstacle_positions, covs_det, covs_inv, self.kinematics, dim_control_points=3,num_body_parts=3, num_samples=num_samples, z_range=(0,2))




    def plan(self,start_pos, end_pos):
        rospy.loginfo("Computing initial guess")
        init_guess = astar_path_spline_fit( start_pos, end_pos, self.obstacle_positions, num_control_points=self.num_control_points, voxel_size=1) 
        rospy.loginfo("Initial guess computed")
        spline = spline_eval((init_guess.reshape(4, self.num_control_points)).T, self.num_samples)

        astar_length = 0
        for i in range(1, self.num_samples):
            astar_length += np.linalg.norm(spline[i,:3] - spline[i-1,:3])
        
        self.ubg[-1] = astar_length * 1.0

    
        rospy.loginfo("Optimizing")
        res = self.solver(x0 = init_guess, lbg = self.lbg, ubg = self.ubg, p = np.concatenate((start_pos, end_pos)))
        self.control_points_opt = np.array(res['x']).reshape(4, self.num_control_points).T
        opt_curve = spline_eval(self.control_points_opt, self.num_samples)
        rospy.loginfo("Optimization done")

        vis = ViserVis()
        vis.add_points(self.obstacle_positions, color = [0,0,1])    
        vis.add_gaussians(self.obstacle_positions, self.obstacle_covs, self.obstacle_colors)
        vis.add_curve(spline[:,:3], color = [1,0,0])
        vis.add_gaussian_path(opt_curve, self.robot_cov ,self.kinematics,color = [0,1,0])
        if vis.show() == 0:
            rospy.loginfo("Visualization rejected, quitting")
            exit(0)

        # # export data for vis testing
        # os.makedirs("test_data", exist_ok=True)

        # np.save("test_data/obstacle_positions.npy", self.obstacle_positions)
        # np.save("test_data/obstacle_covs.npy", self.obstacle_covs)
        # np.save("test_data/spline.npy", spline[:,:3])
        # np.save("test_data/opt_curve.npy", opt_curve)
        # np.save("test_data/robot_cov.npy", self.robot_cov)
        # np.save("test_data/kinematics.npy", self.kinematics)
        # np.save("test_data/means.npy", opt_curve)
        # np.save("test_data/covs.npy", self.robot_cov)

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
 