"""Pure helper functions shared by problem.py and problem_online.py.

Nothing here is specific to static or online environments.  All functions are
stateless and free of side-effects so they can be imported and tested in
isolation.
"""

from __future__ import annotations

from typing import List

import casadi as cas
import numpy as np


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Merge caller-supplied weights with defaults."""
    default = {
        "goal": 1.0,
        "obstacle": 1.0,
        "jerk": 1.0,
        "virtual_jerk": 1.0,
    }
    return default if weights is None else {**default, **weights}


# ---------------------------------------------------------------------------
# Cost terms
# ---------------------------------------------------------------------------


def goal_cost(current_position, target_position, weight: float):
    """Squared distance between current and target task-space positions."""
    return weight * cas.sum1((current_position - target_position) ** 2)


def jerk_cost(
    dddcurve,
    duration,
    real_indices: List[int],
    virtual_indices: List[int],
    real_weight: float,
    virtual_weight: float,
):
    """Integrated squared jerk, per joint, split by real and virtual joints.

        cost = weight * mean_over_joints( integral ||d3q/dt3||^2 dt )

    The quadrature weight is dt = duration / (num_samples - 1).  It must NOT
    be duration**6: `dddcurve` already carries time_scale**3, whose square is
    ((num_control_points - 3) / duration)**6, so a duration**6 factor cancels
    the duration exactly and leaves a bare sum of squared parameter-space
    jerks -- a quantity with no physical units, independent of how long the
    motion takes and proportional to num_samples.

    Because of that, `weights["jerk"]` and `weights["virtual_jerk"]` are in
    different units from before this fix and have to be re-tuned.
    """
    cost = 0.0
    num_samples = dddcurve.shape[0]
    duration_factor = duration / num_samples

    if real_indices:
        real_jerk = dddcurve[:, real_indices]
        cost += (
            real_weight
            * duration_factor
            * cas.sum1(cas.sum2(real_jerk ** 2))
            / len(real_indices)
        )

    if virtual_indices:
        virtual_jerk = dddcurve[:, virtual_indices]
        cost += (
            virtual_weight
            * duration_factor
            * cas.sum1(cas.sum2(virtual_jerk ** 2))
            / len(virtual_indices)
        )

    return cost


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def append_scalar_constraint(
    constraints,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    expr,
    lower: float,
    upper: float,
):
    """Append a single scalar constraint and its bounds."""
    constraints = cas.vertcat(constraints, expr)
    lower_bounds = np.concatenate((lower_bounds, [lower]))
    upper_bounds = np.concatenate((upper_bounds, [upper]))
    return constraints, lower_bounds, upper_bounds


def build_constraints(
    symbolic_type,
    curve,
    start,
    n_dof: int,
    vel_hulls,
    acc_hulls,
    real_indices: List[int],
    virtual_indices: List[int],
    real_wmax: float,
    real_amax: float,
    virtual_wmax: float,
    virtual_amax: float,
):
    """Assemble start-equality, velocity-hull, and acceleration-hull constraints."""
    constraints = symbolic_type([])
    lbg = np.array([], dtype=float)
    ubg = np.array([], dtype=float)

    # Start equality
    for i in range(n_dof):
        constraints, lbg, ubg = append_scalar_constraint(
            constraints, lbg, ubg,
            curve[0, i] - start[i],
            0.0, 0.0,
        )

    # Velocity hull bounds
    for hull in vel_hulls:
        for row in range(hull.shape[0]):
            virtual_vel_cost = 0.0
            for joint_idx in range(hull.shape[1]):
                if joint_idx in virtual_indices:
                    virtual_vel_cost += hull[row, joint_idx] ** 2
                else:
                    constraints, lbg, ubg = append_scalar_constraint(
                        constraints, lbg, ubg,
                        hull[row, joint_idx],
                        -real_wmax, real_wmax,
                    )
            if virtual_indices:
                constraints, lbg, ubg = append_scalar_constraint(
                    constraints, lbg, ubg,
                    virtual_vel_cost,
                    0.0, virtual_wmax ** 2,
                )

    # Acceleration hull bounds
    for hull in acc_hulls:
        for row in range(hull.shape[0]):
            virtual_acc_cost = 0.0
            for joint_idx in range(hull.shape[1]):
                if joint_idx in virtual_indices:
                    virtual_acc_cost += hull[row, joint_idx] ** 2
                else:
                    constraints, lbg, ubg = append_scalar_constraint(
                        constraints, lbg, ubg,
                        hull[row, joint_idx],
                        -real_amax, real_amax,
                    )
            if virtual_indices:
                constraints, lbg, ubg = append_scalar_constraint(
                    constraints, lbg, ubg,
                    virtual_acc_cost,
                    0.0, virtual_amax ** 2,
                )

    return constraints, lbg, ubg


# ---------------------------------------------------------------------------
# Spline / symbolic scaffolding
# ---------------------------------------------------------------------------


def estimate_duration(goal, start_task, vmax: float):
    """Symbolic estimate of trajectory duration from distance and max velocity."""
    duration = cas.norm_2(goal - start_task) / vmax
    return cas.fmax(duration, 1e-3)


def build_collision_point_map(
    robot, q_sym, n_gaussians: int, num_samples: int, curve,
    qdd_sym=None, qdd_curve=None,
):
    """Map robot.collision_points over all samples in parallel.

    qdd_sym/qdd_curve are only passed for robots with requires_acceleration =
    True (e.g. StandardDroneRobot): qdd_curve is the curve's own second
    derivative, sampled the same way as curve, so the mapped function gets
    the exact per-sample acceleration alongside each q.

    Returns
    -------
    mapped_collision_points : shape (3*n_gaussians, num_samples)
    """
    if qdd_sym is not None:
        collision_points_raw = robot.collision_points(q_sym, qdd_sym)
        collision_points_vec = cas.reshape(collision_points_raw.T, 3 * n_gaussians, 1)

        collision_fun = cas.Function("collision_points", [q_sym, qdd_sym], [collision_points_vec])
        collision_map = collision_fun.map(num_samples, "openmp")
        return collision_map(curve.T, qdd_curve.T)

    collision_points_raw = robot.collision_points(q_sym)          # (n_gaussians, 3)
    collision_points_vec = cas.reshape(collision_points_raw.T, 3 * n_gaussians, 1)

    collision_fun = cas.Function("collision_points", [q_sym], [collision_points_vec])
    collision_map = collision_fun.map(num_samples, "openmp")
    return collision_map(curve.T)                                  # (3*n_gaussians, num_samples)


def validate_build_inputs(
    num_control_points: int,
    num_samples: int,
    vmax: float,
) -> None:
    """Raise ValueError for obviously invalid problem dimensions."""
    if num_control_points < 4:
        raise ValueError("num_control_points must be at least 4.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if vmax <= 0:
        raise ValueError("vmax must be positive.")