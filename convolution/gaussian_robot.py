import numpy as np
import casadi as cas
import logging
logging.basicConfig(level =logging.INFO)

# ======================== CLEAN APPROACH ====================================
def create_normal_pdf_functor(dim):
    SYM_TYPE = cas.SX

    x = SYM_TYPE.sym("x", dim, 1)
    mu = SYM_TYPE.sym("mu", dim, 1)
    cov_det = SYM_TYPE.sym("cov_det", 1, 1)
    cov_inv = SYM_TYPE.sym("cov_inv", dim, dim)

    pdf = cas.exp(-0.5 * (x-mu).T@ cov_inv @ (x-mu)) / (cas.sqrt(2 * cas.pi) * cov_det)

    return cas.Function("normal_pdf", [x, mu, cov_det, cov_inv], [pdf])

def create_robot_obstacle_convolution_functor(num_obstacles, dim):
    SYM_TYPE = cas.SX
    pdf_functor = create_normal_pdf_functor(dim)
    
    robot_mean = SYM_TYPE.sym("robot_mean", dim, 1)
    robot_cov = SYM_TYPE.sym("robot_cov", dim * dim)
    obstacle_means = SYM_TYPE.sym("obstacle_means", dim, num_obstacles)
    obstacle_covs = SYM_TYPE.sym("obstacle_covs", dim * dim, num_obstacles)

    obstacle_covs_ = []
    for i in range(obstacle_covs.shape[1]): 
        obstacle_covs_.append(cas.reshape(obstacle_covs[:,i], 3,3))
    robot_cov_ = cas.reshape(robot_cov, 3,3)

    covs_sum = [robot_cov_ + obstacle_cov for obstacle_cov in obstacle_covs_]
    covs_sum_det = [cas.det(cov) for cov in covs_sum]
    covs_sum_inv = [cas.pinv(cov) for cov in covs_sum]
    
    conv = 0

    for i in range(num_obstacles):
        conv += pdf_functor(robot_mean, obstacle_means[:,i], covs_sum_det[i], covs_sum_inv[i])

    norm_conv = conv / num_obstacles
    return cas.Function("robot_obstacle_convolution", [robot_mean, robot_cov, obstacle_means, obstacle_covs], [norm_conv])
 

def create_curve_robot_obstacle_convolution_functor(num_samples, num_obstacles, dim):
    SYM_TYPE = cas.MX

    convolution_functor = create_robot_obstacle_convolution_functor(num_obstacles, dim)

    curve = SYM_TYPE.sym("curve", num_samples, dim)
    robot_cov = SYM_TYPE.sym("robot_cov", dim * dim)
    obstacle_means = SYM_TYPE.sym("obstacle_means", dim, num_obstacles)
    obstacle_covs = SYM_TYPE.sym("obstacle_covs", dim * dim, num_obstacles)
    
    conv = 0
    for i in range(num_samples):
        logging.info(f"Computing convolution for point {i}")
        point = curve[i,:]
        conv += convolution_functor(point, robot_cov, obstacle_means, obstacle_covs)

    return cas.Function("curve_robot_obstacle_convolution", [curve, robot_cov, obstacle_means, obstacle_covs], [conv])
    
