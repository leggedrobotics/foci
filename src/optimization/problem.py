"""Static trajectory optimisation problem builder.

Obstacle statistics are pre-computed from the environment at build time and
uploaded once to the Warp device.  Robot covariances are fixed (not
configuration-dependent).
"""

from __future__ import annotations

from typing import Any

import casadi as cas
import numpy as np

from src.environment.convolution import ConvolutionFunctor
from src.splines.bspline import BSpline
from src.splines.minvo import minvo_hulls
from src.planning.joints import JointGroups

from src.optimization.utils import (
    normalize_weights,
    goal_cost,
    jerk_cost,
    build_constraints,
    build_collision_point_map,
    estimate_duration,
    validate_build_inputs,
)


def _obstacle_cost(
    collision_points,
    environment,
    robot_covariances: np.ndarray,
    num_samples: int,
    weight: float,
):
    """Obstacle cost for a static environment.

    For each robot Gaussian the obstacle + robot covariances are pre-convolved
    once and uploaded to the Warp device.

    Parameters
    ----------
    collision_points:
        Shape (3 * n_gaussians, num_samples).
    environment:
        GaussianEnvironment with fixed obstacle_means and obstacle_covariances.
    robot_covariances:
        Shape (n_gaussians, 3, 3).
    num_samples:
        Number of trajectory samples.
    weight:
        Scalar cost weight.
    """
    robot_covariances = np.asarray(robot_covariances, dtype=float)

    if robot_covariances.ndim != 3 or robot_covariances.shape[1:] != (3, 3):
        raise ValueError("robot_covariances must have shape (n_gaussians, 3, 3).")

    n_gaussians = robot_covariances.shape[0]
    obstacle_means = environment.obstacle_means_at(0)         # static: k is irrelevant
    obstacle_covariances = environment.obstacle_covariances_at(0)

    cost = 0
    callbacks = []

    for g in range(n_gaussians):
        gaussian_points = collision_points[g * 3 : g * 3 + 3, :].T  # (num_samples, 3)

        covs = obstacle_covariances + robot_covariances[g]
        covs_det = np.linalg.det(covs)
        covs_inv = np.linalg.inv(covs)

        convolution = ConvolutionFunctor(
            f"conv_robot_gaussian_{g}",
            num_samples,
            obstacle_means,
            covs_det,
            covs_inv,
        )
        callbacks.append(convolution)
        cost += convolution(gaussian_points)

    return weight * cost / n_gaussians, callbacks


def build_problem(
    robot,
    environment,
    num_control_points: int,
    num_samples: int,
    joint_groups: JointGroups,
    weights: dict[str, float] | None = None,
    vmax: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list]:
    """Build the static trajectory optimisation NLP.

    Returns
    -------
    nlp, lbg, ubg, callbacks
    """
    validate_build_inputs(num_control_points, num_samples, vmax)

    weights = normalize_weights(weights)

    symbolic_type = cas.MX
    n_dof = robot.n_dof

    joint_groups.validate(n_dof)
    real_indices = joint_groups.real_indices(n_dof)
    virtual_indices = joint_groups.virtual_indices

    # --- Decision variables and parameters ---------------------------
    control_points = symbolic_type.sym("control_points", num_control_points, n_dof)
    decision_variables = cas.vertcat(cas.vec(control_points))

    start = symbolic_type.sym("start", n_dof, 1)
    goal  = symbolic_type.sym("goal",  3,     1)
    params = cas.vertcat(cas.vec(start), cas.vec(goal))

    # --- Spline and time scaling -------------------------------------
    bspline = BSpline(control_points)
    curve   = bspline.spline_eval(num_samples)

    start_task         = robot.f_task(curve[0, :])
    estimated_duration = estimate_duration(goal, start_task, vmax)
    time_scale         = (num_control_points - 3) / estimated_duration

    dddcurve = time_scale ** 3 * bspline.spline_eval(num_samples, derivative_order=3)

    # --- Collision points --------------------------------------------
    q_sym = symbolic_type.sym("q", n_dof)
    n_gaussians = int(robot.collision_points(q_sym).shape[0])

    # Robots whose collision geometry depends on acceleration (e.g. a
    # standard drone's acceleration-coupled tilt) get the curve's own exact
    # second derivative -- same family as acc_hulls below, but evaluated
    # pointwise at each sample instead of bounded over a control interval.
    if robot.requires_acceleration:
        qdd_sym = symbolic_type.sym("qdd", n_dof)
        ddcurve = time_scale ** 2 * bspline.spline_eval(num_samples, derivative_order=2)
    else:
        qdd_sym = None
        ddcurve = None

    mapped_collision_points = build_collision_point_map(
        robot, q_sym, n_gaussians, num_samples, curve,
        qdd_sym=qdd_sym, qdd_curve=ddcurve,
    )

    # --- Constraints -------------------------------------------------
    vel_hulls = [time_scale      * h for h in minvo_hulls(control_points, derivative_order=1)]
    acc_hulls = [time_scale ** 2 * h for h in minvo_hulls(control_points, derivative_order=2)]

    constraints, lbg, ubg = build_constraints(
        symbolic_type=symbolic_type,
        curve=curve,
        start=start,
        n_dof=n_dof,
        vel_hulls=vel_hulls,
        acc_hulls=acc_hulls,
        real_indices=real_indices,
        virtual_indices=virtual_indices,
        real_wmax=joint_groups.real_wmax,
        real_amax=joint_groups.real_amax,
        virtual_wmax=joint_groups.virtual_wmax,
        virtual_amax=joint_groups.virtual_amax,
    )

    # --- Costs -------------------------------------------------------
    cost_goal, cost_jerk, (cost_obstacles, callbacks) = (
        goal_cost(robot.f_task(curve[-1, :]), goal, weight=weights["goal"]),
        jerk_cost(
            dddcurve, estimated_duration,
            real_indices=real_indices, virtual_indices=virtual_indices,
            real_weight=weights["jerk"], virtual_weight=weights["virtual_jerk"],
        ),
        _obstacle_cost(
            mapped_collision_points, environment,
            robot.collision_covariances(),
            num_samples=num_samples, weight=weights["obstacle"],
        ),
    )

    # --- NLP ---------------------------------------------------------
    nlp = {
        "x": decision_variables,
        "f": cost_goal + cost_obstacles + cost_jerk,
        "p": params,
        "g": constraints,
    }

    return nlp, lbg, ubg, callbacks