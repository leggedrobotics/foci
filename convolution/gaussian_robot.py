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
    cov_inv = SYM_TYPE.sym("cov_inv", dim , dim )


    pdf = cas.exp(-0.5 * (x-mu).T@ cov_inv@ (x-mu)) / (cas.sqrt(2 * cas.pi) * cov_det)

    print("cov_inv shape",cov_inv.shape)

    return cas.Function("normal_pdf", [x, mu, cov_det, cov_inv], [pdf])

def create_matrix_inverse_functor(dim):
    SYM_TYPE = cas.SX

    cov = SYM_TYPE.sym("cov", dim, dim)
    cov_inv = cas.pinv(cov)

    return cas.Function("matrix_inverse", [cov], [cov_inv])

def create_matrix_det_functor(dim):
    SYM_TYPE = cas.SX

    cov = SYM_TYPE.sym("cov", dim, dim)
    cov_det = cas.det(cov)

    return cas.Function("matrix_det", [cov], [cov_det])


def create_robot_obstacle_convolution_functor(num_obstacles, dim, parallelization = "openmp"):
    SYM_TYPE = cas.SX
    pdf_functor_map = create_normal_pdf_functor(dim).map(num_obstacles, parallelization)
    inverse_functor_map = create_matrix_inverse_functor(dim).map(num_obstacles, parallelization)
    det_functor_map = create_matrix_det_functor(dim).map(num_obstacles, parallelization)
    
    robot_mean = SYM_TYPE.sym("robot_mean", dim, 1)
    robot_cov = SYM_TYPE.sym("robot_cov", dim, dim)
    obstacle_means = SYM_TYPE.sym("obstacle_means", dim, num_obstacles)
    obstacle_covs = SYM_TYPE.sym("obstacle_covs", dim, dim *  num_obstacles)


    covs_sum = obstacle_covs + robot_cov 
    covs_sum_det = det_functor_map(covs_sum)
    covs_sum_inv = inverse_functor_map(covs_sum)

    print("covs_sum_inv shape",covs_sum_inv.shape) 
    print("covs_sum_det shape",covs_sum_det.shape)  
    
    
    pdf_evals = pdf_functor_map(robot_mean, obstacle_means, covs_sum_det, covs_sum_inv)
    norm_conv = cas.sum2(pdf_evals) / num_obstacles

    return cas.Function("robot_obstacle_convolution", [robot_mean, robot_cov, obstacle_means, obstacle_covs], [norm_conv])
 

def create_curve_robot_obstacle_convolution_functor(num_samples, num_obstacles, dim, parallelization = "openmp"):
    SYM_TYPE = cas.MX

    convolution_functor = create_robot_obstacle_convolution_functor(num_obstacles, dim, parallelization = parallelization)
    
    curve = SYM_TYPE.sym("curve", dim, num_samples)
    robot_cov = SYM_TYPE.sym("robot_cov", dim , dim)
    obstacle_means = SYM_TYPE.sym("obstacle_means", dim, num_obstacles)
    obstacle_covs = SYM_TYPE.sym("obstacle_covs", dim, dim *  num_obstacles)


#    convolution_functor_with_set_parameters = cas.Function("robot_obstacle_convolution", [point], [convolution_functor(point, robot_cov, obstacle_means, obstacle_covs)])

    convolution_map = convolution_functor.map(num_samples, parallelization)
    out_map = convolution_map(curve, robot_cov, obstacle_means, obstacle_covs)
    print("out_map shape",out_map.shape)

    out = cas.sum2(out_map)/ num_samples # Does that change solver speed?
    print("outshape",out.shape)   
    return cas.Function("robot_obstacle_convolution", [curve, robot_cov, obstacle_means, obstacle_covs], [out])
