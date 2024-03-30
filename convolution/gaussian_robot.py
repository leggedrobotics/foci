import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import os
import logging
logging.basicConfig(level =logging.INFO)


def normal_pdf_cas(x, mean, cov, cov_determinat = None, cov_inverse = None):
    """Return the evaluation of the multivariate normal distribution at x
    @args:
        x : cas.SX
        mean: np.array
        cov: np.array
    """
    if cov_determinat is None:
        cov_determinat = np.linalg.det(cov)
    if cov_inverse is None:
        cov_inverse = np.linalg.inv(cov)

    return 1/np.sqrt((2*np.pi)**2 * cov_determinat)* cas.exp(-0.5 * (x - mean).T @cov_inverse@ (x - mean))

def robot_obstacle_convolution(robot_mean, robot_cov, obstacle_means, obstacle_covs, obstacle_cov_dets = None, obstacle_cov_invs= None):
    """Return the convolution of the robot and obstacle normal distributions
    @args
        robot_mean: cas.SX
        robot_cov: np.array
        obstacle_means: [np.array]
        obstacle_covs: [np.array]
    """
    sum = 0
    if obstacle_cov_det is None or obstacle_cov_inv is None:
        for obstacle_mean, obstacle_cov in zip(obstacle_means, obstacle_covs):
            sum += normal_pdf_cas(robot_mean, obstacle_mean, robot_cov + obstacle_cov)

    else:
        for obstacle_mean, obstacle_cov in zip(obstacle_means, obstacle_covs, obstacle_cov_dets, obstacle_cov_invs):
            sum += normal_pdf_cas(robot_mean, obstacle_mean, robot_cov + obstacle_cov)
    # instead of summing the convolutions we can also take the maximum
    return sum

def curve_robot_obstacle_convolution(curve, robot_cov, obstacle_means, obstacle_covs):
    """Return the convolution of the robot and obstacle normal distributions
    @args
        curve: np.array
        robot_cov: np.array
        obstacle_means: [np.array]
        obstacle_covs: [np.array]
    """
    covs_sum = [robot_cov + obstacle_cov for obstacle_cov in obstacle_covs]
    covs_sum_det = [np.linalg.det(cov) for cov in covs_sum]
    covs_sum_inv = [np.linalg.inv(cov) for cov in covs_sum]

    # define casadi array
    pdf_eval = cas.SX([])

    print(curve.shape)
    for i in range(curve.shape[0]):
        point = curve[i,:].T
        logging.info("Build for point " + str(i))
        for obstacle_mean, cov_sum, cov_sum_det, cov_sum_inv in zip(obstacle_means, covs_sum, covs_sum_det, covs_sum_inv):
            pdf_eval = cas.vertcat(
                pdf_eval, normal_pdf_cas(point, obstacle_mean, cov_sum, cov_determinat = cov_sum_det, cov_inverse = cov_sum_inv))

    return cas.norm_2(pdf_eval) / pdf_eval.shape[0]


