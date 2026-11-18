"""Shared metrics for the planner benchmarks.

Everything a benchmark script compares across planners lives here, so the
three planners are measured with one function instead of each reporting its
own internal cost in its own units.

Resolution is part of every derivative metric here
--------------------------------------------------
`jerk_metric` and `limit_scaling_factor` both resample to `num_samples` and
then take finite differences.  A planner's output is a POLYLINE, so its second
and third derivatives are impulses at the corners and any finite-difference
estimate of them grows without bound as `num_samples` grows.  Measured on a
12-waypoint trajectory:

    num_samples   12     25     50    100    200
    max|acc|    17.1   20.0   43.9   93.0  170.0

Every planner in a comparison must therefore be measured at the SAME
`num_samples`, and that value must be the coarsest representation involved
(a waypoint planner has nothing finer to offer).  Pass the benchmark's
`num_waypoints` explicitly -- never rely on the default.
"""

from time import perf_counter

import numpy as np


_EPS = 1e-12


def median_solve_time(fn, repeats: int = 3):
    """
    Call `fn()` `repeats` times and return (last result, median wall time).

    Ported from the FOCI paper's eval/baselines.py (`_median_time`). A single
    timed call conflates the planner's steady-state solve time with one-off
    costs that are not representative of it: IPOPT/CasADi codegen on the
    first build, and the warp/CUDA kernels behind the obstacle-cost
    convolution (ConvolutionFunctor) being JIT-compiled on their first
    launch in the process. A planner must therefore be called once as an
    untimed warmup before any of these repeats -- the paper eval does this
    explicitly ("warmup: warp kernel compile must not count as solve time").

    This does mean the planner's `.plan()` runs `repeats + 1` times instead
    of once, so a benchmark script using this is correspondingly slower.
    """
    fn()  # warmup, not timed

    times = []
    result = None

    for _ in range(repeats):
        t0 = perf_counter()
        result = fn()
        times.append(perf_counter() - t0)

    return result, float(np.median(times))


def scene_size(obstacle_means) -> float:
    """
    XY diagonal of the obstacle point cloud's bounding box.

    Used to report path length as a dimensionless ratio (length / scene_size)
    comparable across scenes of different physical size, matching the FOCI
    paper's eval/baselines.py `_score()` (`lengths.append(s[-1] / scene_size)`).
    The current benchmarks only ever reported the raw, scene-size-dependent
    length.
    """
    means = np.asarray(obstacle_means, dtype=float)
    extent = means.max(axis=0) - means.min(axis=0)

    return float(np.hypot(extent[0], extent[1]))


def resample_trajectory(trajectory, num_samples: int) -> np.ndarray:
    """
    Resample a waypoint trajectory to `num_samples` points by linear
    interpolation in the normalised parameter [0, 1].

    Planners return trajectories at different resolutions (12 waypoints for
    STOMP/CHOMP, 25 spline samples for FOCI), and every path/derivative metric
    is resolution dependent, so metrics must be computed after resampling all
    of them to a common number of points.
    """
    trajectory = np.asarray(trajectory, dtype=float)

    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be a 2D array of shape (N, D), got {trajectory.shape}."
        )
    if trajectory.shape[0] < 2:
        raise ValueError(
            f"trajectory must contain at least two points, got {trajectory.shape[0]}."
        )
    if num_samples < 2:
        raise ValueError(f"num_samples must be at least 2, got {num_samples}.")
    if trajectory.shape[0] == num_samples:
        return trajectory.copy()

    old_s = np.linspace(0.0, 1.0, trajectory.shape[0])
    new_s = np.linspace(0.0, 1.0, num_samples)

    return np.vstack(
        [np.interp(new_s, old_s, trajectory[:, d]) for d in range(trajectory.shape[1])]
    ).T


_MIN_DERIVATIVE_SAMPLES: int = 3


def _central_derivative(xi: np.ndarray, dt: float, order: int) -> np.ndarray:
    """Central finite differences of a waypoint trajectory, edges replicated.

    Needs at least three rows: a central difference consumes one row at each
    end, and with two rows `inner` comes out empty, which collapses the result
    to shape (0, D) and every later `.max()` raises "zero-size array to
    reduction operation" from somewhere unrelated.  Fail here instead.
    """
    d = np.asarray(xi, dtype=float)

    if d.shape[0] < _MIN_DERIVATIVE_SAMPLES:
        raise ValueError(
            f"a central derivative needs at least {_MIN_DERIVATIVE_SAMPLES} "
            f"samples, got {d.shape[0]}."
        )

    for _ in range(order):
        inner = (d[2:] - d[:-2]) / (2.0 * max(dt, _EPS))
        d = np.vstack([inner[:1], inner, inner[-1:]])

    return d


def jerk_metric(trajectory, duration: float, num_samples: int = 200) -> float:
    """
    Integrated squared jerk of a trajectory, in physical units.

        J = integral ||d3q/dt3||^2 dt

    Computed after resampling to `num_samples` points, and the resolution is
    part of the metric: a coarse polyline underestimates the jerk of the
    smooth curve it samples (at 12 waypoints a 4-point stencil recovers only
    about 22% of the true value), so this is a comparison figure, not an
    absolute physical one.

    For that comparison to be fair, every planner must be measured at the SAME
    `num_samples`, which in practice means the coarsest representation
    involved -- a waypoint planner has nothing finer to offer, and upsampling
    its polyline would just put all the curvature in the corners.  Pass the
    benchmark's `num_waypoints`.

    This is deliberately NOT any planner's internal cost: those are in
    different units and are not comparable with each other.
    """
    if num_samples < _MIN_DERIVATIVE_SAMPLES:
        raise ValueError(
            f"num_samples must be at least {_MIN_DERIVATIVE_SAMPLES}, got "
            f"{num_samples}."
        )

    xi = resample_trajectory(trajectory, num_samples)
    dt = float(duration) / max(num_samples - 1, 1)
    jerk = _central_derivative(xi, dt, order=3)

    return float(dt * np.sum(jerk ** 2))


def limit_scaling_factor(
    trajectory,
    duration: float,
    joint_groups,
    n_dof: int,
    num_samples: int = 200,
) -> tuple[float, float]:
    """
    Smallest uniform time scaling that makes a trajectory respect the
    velocity and acceleration limits.

    Scaling the duration by s divides velocity by s and acceleration by s^2,
    and leaves the geometry untouched, so the factor is exact in one shot:

        s = max( max|v| / wmax , sqrt(max|a| / amax) , 1 )

    Planners that do not enforce the limits (CHOMP does not look at them at
    all, and STOMP only adds a soft penalty whose relative weight is
    arbitrary) can then be compared on equal footing: the returned duration is
    how long each one actually needs to be executable.

    `num_samples` is part of the metric, exactly as in `jerk_metric`: the
    acceleration of a polyline is an impulse at every corner, so max|acc| --
    and with it the returned scale -- grows with the resampling resolution.
    Always pass the benchmark's `num_waypoints`; leaving the default 200 on a
    12-waypoint trajectory inflates the scale by roughly 3x (see the module
    docstring).

    Returns
    -------
    (scale, feasible_duration)
        `scale` is 1.0 when the trajectory already satisfies the limits.
    """
    if num_samples < _MIN_DERIVATIVE_SAMPLES:
        raise ValueError(
            f"num_samples must be at least {_MIN_DERIVATIVE_SAMPLES}, got "
            f"{num_samples}."
        )

    xi = resample_trajectory(trajectory, num_samples)
    dt = float(duration) / max(num_samples - 1, 1)

    vel = _central_derivative(xi, dt, order=1)
    acc = _central_derivative(xi, dt, order=2)

    real_idx = list(joint_groups.real_indices(n_dof))
    virt_idx = list(joint_groups.virtual_indices)

    scales = [1.0]

    if real_idx:
        scales.append(np.abs(vel[:, real_idx]).max() / max(joint_groups.real_wmax, _EPS))
        scales.append(
            np.sqrt(np.abs(acc[:, real_idx]).max() / max(joint_groups.real_amax, _EPS))
        )

    if virt_idx:
        # Virtual joints are limited by the norm of their sub-vector, matching
        # the constraint FOCI imposes on the MINVO hulls.
        scales.append(
            np.linalg.norm(vel[:, virt_idx], axis=1).max()
            / max(joint_groups.virtual_wmax, _EPS)
        )
        scales.append(
            np.sqrt(
                np.linalg.norm(acc[:, virt_idx], axis=1).max()
                / max(joint_groups.virtual_amax, _EPS)
            )
        )

    scale = float(max(scales))

    return scale, scale * float(duration)


def path_length(trajectory) -> float:
    """
    Compute the length of a trajectory as the sum of the Euclidean
    distances between consecutive points.

    Works for any sequence of points (joint-space configurations,
    task-space positions, control points, ...), as long as it can be
    interpreted as an array of shape (N, D).

    Parameters
    ----------
    trajectory :
        Sequence of N points of dimension D (array-like of shape (N, D)).

    Returns
    -------
    float
        Total path length, i.e. sum_{i=0}^{N-2} || trajectory[i+1] - trajectory[i] ||.
        Returns 0.0 if the trajectory has fewer than 2 points.
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be a 2D array of shape (N, D), got shape {trajectory.shape}."
        )

    if trajectory.shape[0] < 2:
        return 0.0

    segment_vectors = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    return float(np.sum(segment_lengths))


def path_lengths(trajectories: dict) -> dict:
    """
    Compute the path length of several trajectories at once.

    Parameters
    ----------
    trajectories :
        Mapping from a name (e.g. planner name) to its trajectory.

    Returns
    -------
    dict
        Mapping from the same names to their path length.
    """
    return {name: path_length(trajectory) for name, trajectory in trajectories.items()}


def _resample_by_arc_length(trajectory: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Resample a polyline to `num_samples` points equally spaced BY ARC
    LENGTH (not by index/parameter), matching the FOCI paper's
    eval/baselines.py resampling before measuring clearance:

        s = cumsum of consecutive-point distances
        xyz = interp over linspace(0, s[-1], N_DENSE)

    Unlike `resample_trajectory` (uniform in the parameter index -- fine
    for a B-spline's own parameter, but not a substitute for this), this
    makes the clearance check density-independent of how many points the
    planner's native output happens to have.  The paper's curves are
    task-space xyz; here `trajectory` is joint-space, so this resamples by
    arc length in CONFIGURATION space -- the direct analogue, and correct
    for the thing it feeds (robot.collision_points(q) is per-configuration
    anyway, so there is no task-space curve to resample before that).
    """
    trajectory = np.asarray(trajectory, dtype=float)
    seg_lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    if s[-1] == 0.0:
        return np.tile(trajectory[0], (num_samples, 1))

    new_s = np.linspace(0.0, s[-1], num_samples)

    return np.vstack(
        [np.interp(new_s, s, trajectory[:, d]) for d in range(trajectory.shape[1])]
    ).T


def minimum_robot_environment_distance(
    robot,
    environment,
    trajectory,
    opacities: np.ndarray | None = None,
    opacity_min: float = 0.5,
    sigma: float = 1.0,
    n_neighbours: int = 96,
    num_dense_samples: int = 400,
    include_robot_covariance: bool = True,
):
    """
    Smallest surface-to-surface clearance between any robot Gaussian and any
    obstacle Gaussian along the trajectory, at the `sigma` level.

    This used to report plain centre-to-centre distance between Gaussian
    means, which is not a clearance at all -- a robot Gaussian sitting right
    at an obstacle's 1-sigma boundary and one still a metre away could report
    the same number, since neither Gaussian's extent was taken into account.

    With `include_robot_covariance=True` (default), the two Gaussians are
    combined via a Minkowski sum (combined_cov = robot_cov + obstacle_cov),
    exactly matching the obstacle-avoidance cost every planner in this repo
    already optimises against (see ConvolutionFunctor's callers: problem.py,
    chomp_planner.py, stomp_planner.py all build
    `obstacle_covariances + robot_covariances[g]`). The reported distance is
    from the robot Gaussian's centre to the `sigma`-level surface of that
    combined ellipsoid centred at the obstacle's mean, found by bisection on
    the ellipsoid's principal axes. A negative value means the robot centre
    is already inside the combined ellipsoid (penetration estimate, not
    exact).

    With `include_robot_covariance=False`, the robot's own extent is
    dropped (combined_cov = obstacle_cov alone) -- this is what the FOCI
    paper's eval/baselines.py Environment.surface_distance literally
    measures: distance from the bare path centreline to the obstacle's
    surface, with the robot's own size never subtracted from the reported
    number at all (its baselines used a fixed-radius sphere robot whose
    radius only ever fed the PLANNER's clearance constraint, never the
    reported safety metric). True means a smaller (more conservative)
    number than the paper's own metric would report for the same path --
    set this False to replicate the paper's numbers exactly instead.

    `opacities`, when given (shape (n_obstacles,) or (n_obstacles, 1)),
    drops obstacles with opacity <= `opacity_min` before measuring, matching
    the paper eval's OPACITY_MIN=0.5 -- a hallucinated near-transparent splat
    should not count as a safety violation.

    Nearest-neighbour candidates per (sample, robot Gaussian) are found with
    a KD-tree (`n_neighbours`, matching the paper eval's 96) instead of
    testing every obstacle: the previous centre-distance version vectorised
    over all obstacles at once, which does not extend to a per-obstacle
    bisection (60 iterations each) without becoming the triple Python loop
    this function was rewritten to avoid in the first place.

    `trajectory` is resampled to `num_dense_samples` points equally spaced
    BY ARC LENGTH (not by its native parameter index) before measuring,
    matching the paper eval's N_DENSE=400 -- without this, a trajectory
    with few native points could pass close to an obstacle BETWEEN two of
    them without any sample ever landing near enough to detect it, and
    different planners being compared at different native resolutions
    (e.g. a 25-point FOCI/CHOMP output against a 12-point STOMP one) is not
    even measuring them at the same density to begin with. This changes
    the reported minimum vs reading it directly off a planner's own
    coarse output -- it does not loosen or tighten any actual clearance.
    """
    from scipy.spatial import cKDTree

    trajectory = _resample_by_arc_length(trajectory, num_dense_samples)

    obstacle_means = np.asarray(environment.obstacle_means, dtype=float)
    obstacle_covs = np.asarray(environment.obstacle_covariances, dtype=float)

    if opacities is not None:
        keep = np.asarray(opacities, dtype=float).reshape(-1) > opacity_min
        kept_indices = np.flatnonzero(keep)
        obstacle_means = obstacle_means[keep]
        obstacle_covs = obstacle_covs[keep]
    else:
        kept_indices = None

    if obstacle_means.shape[0] == 0:
        return np.inf, None

    tree = cKDTree(obstacle_means)
    robot_covs = np.asarray(robot.collision_covariances(), dtype=float)  # (G, 3, 3)

    # Conservative candidate radius per robot Gaussian: sqrt(max eigenvalue)
    # of the obstacle covariances (computed once, not per query) plus this
    # robot Gaussian's own, so the KD-tree ball cannot miss the true nearest
    # surface -- it may include extra, harmless candidates instead.
    obstacle_max_eig = float(np.linalg.eigvalsh(obstacle_covs).max())
    robot_max_eig = np.linalg.eigvalsh(robot_covs).max(axis=1)  # (G,)
    k = min(n_neighbours, obstacle_means.shape[0])

    min_dist = np.inf
    argmin = None

    for t, q in enumerate(trajectory):
        robot_points = np.asarray(robot.collision_points(q), dtype=float).reshape(-1, 3)

        for g, p in enumerate(robot_points):
            radius = sigma * (np.sqrt(obstacle_max_eig) + np.sqrt(robot_max_eig[g])) + 1e-3
            idx = tree.query_ball_point(p, radius)

            if not idx:
                _, nearest = tree.query(p, k=1)
                idx = [int(nearest)]
            elif len(idx) > k:
                dists, order = tree.query(p, k=k)
                idx = np.asarray(idx)[np.argsort(
                    np.linalg.norm(obstacle_means[idx] - p, axis=1)
                )[:k]].tolist()

            idx = np.asarray(idx)
            combined = (
                obstacle_covs[idx] + robot_covs[g]
                if include_robot_covariance
                else obstacle_covs[idx]
            )  # (K, 3, 3)
            eigvals, rots = np.linalg.eigh(combined)  # (K, 3), (K, 3, 3)
            stds = np.sqrt(np.maximum(eigvals, 1e-12))

            diff = p - obstacle_means[idx]  # (K, 3)
            z = np.abs(np.einsum("kji,kj->ki", rots, diff))  # coords in principal axes
            a = sigma * stds  # semi-axes of the combined ellipsoid

            m = np.sqrt(((z / a) ** 2).sum(-1))
            z_c, a2 = np.maximum(z, 1e-12), a ** 2
            # Bracket of the root of f(s) = sum (a*z / (s + a^2))^2 - 1,
            # decreasing in s (same construction as the paper eval).
            lo = -a2.min(-1) + np.sqrt(a2.min(-1)) * np.take_along_axis(
                z_c, a2.argmin(-1)[..., None], -1
            )[..., 0]
            hi = np.linalg.norm(a * z_c, axis=-1)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                pos = (((a * z_c) / (mid[..., None] + a2)) ** 2).sum(-1) > 1.0
                lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
            dist = np.linalg.norm(z_c - a2 * z_c / (0.5 * (lo + hi)[..., None] + a2), axis=-1)
            dist = np.where(m < 1.0, -(1.0 - m) * a.min(-1), dist)  # inside: penetration estimate

            j_local = int(np.argmin(dist))
            d = float(dist[j_local])

            if d < min_dist:
                obstacle_idx = int(idx[j_local])
                min_dist = d
                argmin = {
                    "sample": t,
                    "robot_gaussian": g,
                    "obstacle": int(kept_indices[obstacle_idx]) if kept_indices is not None else obstacle_idx,
                    "robot_point": p,
                    "obstacle_point": obstacle_means[obstacle_idx],
                }

    return min_dist, argmin