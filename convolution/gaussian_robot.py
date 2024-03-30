import numpy as np
import casadi as cas
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
    conv = 0
    if obstacle_cov_dets is None or obstacle_cov_invs is None:
        for obstacle_mean, obstacle_cov in zip(obstacle_means, obstacle_covs):
            conv += normal_pdf_cas(robot_mean, obstacle_mean, robot_cov + obstacle_cov)

    else:
        for obstacle_mean, obstacle_cov in zip(obstacle_means, obstacle_covs, obstacle_cov_dets, obstacle_cov_invs):
            conv += normal_pdf_cas(robot_mean, obstacle_mean, robot_cov + obstacle_cov)
    # instead of summing the convolutions we can also take the maximum
    return conv

def curve_robot_obstacle_convolution(curve, robot_cov, obstacle_means, obstacle_covs):
    """Return the convolution of the robot and obstacle normal distributions
    @args
        curve: np.array
        robot_cov: np.array
        obstacle_means: [np.array]
        obstacle_covs: [np.array]
    """
    obstacle_covs_ = []
    for i in range(obstacle_covs.shape[1]):
    
        obstacle_covs_.append(cas.reshape(obstacle_covs[:,i], 3,3))
    robot_cov = cas.reshape(robot_cov, 3,3)

    covs_sum = [robot_cov + obstacle_cov for obstacle_cov in obstacle_covs_]
    covs_sum_det = [cas.det(cov) for cov in covs_sum]
    covs_sum_inv = [cas.pinv(cov) for cov in covs_sum]

   

    # define casadi array
    pdf_eval = cas.SX([])

    print(curve.shape)
    for i in range(curve.shape[0]):
        point = curve[i,:].T
        logging.info("Build for point " + str(i))
        for j in range(obstacle_means.shape[1]):
            obstacle_mean = obstacle_means[:,j]
            cov_sum = covs_sum[j]
            cov_sum_det = covs_sum_det[j]
            cov_sum_inv = covs_sum_inv[j]

            pdf_eval = cas.vertcat(
                pdf_eval, normal_pdf_cas(point, obstacle_mean, cov_sum, cov_determinat = cov_sum_det, cov_inverse = cov_sum_inv))

    return cas.norm_2(pdf_eval) / pdf_eval.shape[0]


