"""
STOMP Planner
=============

Stochastic Trajectory Optimization for Motion Planning.

Follows the same interface as CHOMPPlanner and Planner so it can be used
as a drop-in replacement anywhere a PlanningResult is expected.

Algorithm summary
-----------------
STOMP does NOT use gradients. Instead, at each iteration it:
  1. Samples K noisy trajectory perturbations  δξ_k ~ N(0, R^{-1})
     where R is the smoothness precision matrix.
  2. Evaluates the total cost Q_k for each noisy trajectory ξ + δξ_k.
  3. Computes per-waypoint probability weights w_k ∝ exp(-h · Q_k).
  4. Updates the mean trajectory:  Δξ = R^{-1} Σ_k w_k δξ_k
     (the update is automatically smooth because R^{-1} filters it).
  5. Re-pins the endpoints.

Cost terms (all evaluated numerically, no CasADi graph at solve time):
  - Obstacle cost   : Gaussian convolution, same as problem.py / CHOMPPlanner.
  - Jerk cost       : penalises the 3rd finite difference of the trajectory,
                      matching the _jerk_cost() logic in problem.py.
  - Constraint cost : soft penalty for velocity, acceleration, and joint-limit
                      violations (magnitude of the violation, not a hard bound).

References
----------
Kalakrishnan et al., "STOMP: Stochastic Trajectory Optimization for
Motion Planning", ICRA 2011.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import casadi as cas
import numpy as np

from src.benchmark.utils import limit_scaling_factor
from src.environment.convolution import ConvolutionFunctor
from src.planning.joints import JointGroups
from src.planning.result import PlanningResult


_EPS = 1e-10


# =============================================================================
# Smoothness precision matrix  R  and its inverse
# =============================================================================

def _build_smoothness_matrix(num_waypoints, dt):
    """
    Build the finite-difference precision matrix R = A^T A / dt^2.

    A is the (T+2, T) second-difference operator of Kalakrishnan et al., i.e.
    it INCLUDES the two boundary rows at each end, which reference waypoints
    outside the trajectory and are therefore truncated.  Those rows are what
    makes A full column rank.

    Dropping them (a (T-2, T) operator with interior rows only) leaves A^T A
    with a 2-dimensional nullspace -- constants and ramps -- so any ridge
    added to invert it becomes the dominant term of R^{-1}, and the sampled
    noise degenerates into a rigid affine shift of the whole trajectory whose
    per-waypoint sigma is LARGEST at the pinned endpoints.  With the boundary
    rows, cond(R) drops from ~5e8 to ~1e3 and the noise recovers the
    bell-shaped profile that lets STOMP deform one part of the path.
    """
    T = num_waypoints
    A = np.zeros((T + 2, T))

    for i in range(T + 2):
        for offset, value in ((0, 1.0), (1, -2.0), (2, 1.0)):
            j = i - 2 + offset
            if 0 <= j < T:
                A[i, j] = value

    # No ridge needed: A has full column rank, so A^T A is already invertible.
    return (A.T @ A) / max(dt ** 2, _EPS)

def _build_update_matrix(R_inv: np.ndarray) -> np.ndarray:
    """
    STOMP's update projection M (Kalakrishnan et al. 2011, sec. III):

        M = R^-1, with each column scaled so its largest element is 1/N

    and the update is  delta_xi = M @ sum_k w_k * eps_k, not the bare sum.

    Why it is needed: each eps_k is drawn from N(0, R^-1) and so is smooth on
    its own, but the probability-weighted SUM is not -- the weights depend on
    the sampled costs, so the average leaves the smooth subspace the samples
    came from.  M is the low-pass filter that puts it back, and it is the
    reason the paper's updates keep a trajectory smooth.  Without it nothing
    constrains the high-frequency content of the step, and the result shows
    visible kinks (worst at the endpoints, where the finite-difference jerk
    term is blind).

    Why the column normalisation matters, and why dropping M was a misdiagnosis:
    a previous version removed the projection because `R_inv @ delta` exploded
    the step ("row sums ~1,000,000").  Measured, that gain belonged to the
    SINGULAR (T-2, T) smoothness matrix plus its 1e-6 ridge -- ||R_old^-1||_inf
    is 1.6e6 for every T -- which commit 26b73fe already fixed.  With the
    full-rank (T+2, T) operator, ||R^-1||_inf is 3.2 (T=12), 27.6 (T=15),
    5.3 (T=40), and the paper's normalisation bounds the row sums by 1, giving
    ||M||_inf <= 0.94.  So the step stays the size of the noise that produced
    it, and the update comes out ~260x smoother (normalised second differences
    5.00 -> 0.019 at T=12).
    """
    n = R_inv.shape[0]
    M = np.array(R_inv, dtype=float, copy=True)
    column_max = M.max(axis=0)
    return M / np.maximum(column_max, _EPS) / float(n)


def _build_precision_inverse(R: np.ndarray) -> np.ndarray:
    """
    Return R^{-1}.

    R^{-1} is the covariance of the noise distribution and the smoothing
    kernel for trajectory updates.  Computed once and reused across all
    iterations and DOFs.
    """
    return np.linalg.inv(R)


# =============================================================================
# Noise sampling
# =============================================================================

def _sample_noise(
    R_inv: np.ndarray,
    n_dof: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw K smooth noise trajectories with unit std, ready to be scaled by
    noise_scale in the caller.

    Returns
    -------
    noise : (K, T, n_dof)
        Each noise[k] is a smooth perturbation. std ≈ 1 per waypoint so
        that noise_scale has direct physical meaning (metres, radians, etc).

    Why normalize L?
    ----------------
    The raw Cholesky factor L = chol(R^{-1}) amplifies white noise z by
    sqrt(diag(R^{-1})).  For typical STOMP parameters (T=40, dt=1/39),
    diag(R^{-1}) ≈ 25,000, so L amplifies by ~158x.  Without normalization,
    noise_scale=0.1 produces noise with std≈15.8 — catastrophically large
    for a drone flying over a few metres.

    Dividing L by L_scale = sqrt(max diag(R^{-1})) makes the resulting
    noise have std≈1 regardless of T and dt, so noise_scale directly
    controls the perturbation magnitude in the robot's configuration units.
    The smoothness shape of the noise (low-frequency bias) is preserved
    because we scale the whole matrix uniformly.
    """
    T = R_inv.shape[0]
    L = np.linalg.cholesky(R_inv)                        # (T, T)
    L_scale = np.sqrt(np.diag(R_inv).max())              # scalar ≈ 158 for T=40
    L_norm  = L / max(float(L_scale), _EPS)              # unit-scale Cholesky

    z     = rng.standard_normal(size=(n_samples, T, n_dof))  # (K, T, n_dof)
    noise = np.einsum("ij,kjd->kid", L_norm, z)              # (K, T, n_dof)
    return noise


# =============================================================================
# Finite-difference derivatives  (pure numpy, no CasADi)
# =============================================================================

def _fd_velocity(xi: np.ndarray, dt: float) -> np.ndarray:
    """
    First derivative via central differences.  Shape: (T, n_dof).
    Endpoints use forward/backward differences.
    """
    vel = np.zeros_like(xi)
    vel[1:-1] = (xi[2:] - xi[:-2]) / (2.0 * max(dt, _EPS))
    vel[0]    = (xi[1]  - xi[0])   / max(dt, _EPS)
    vel[-1]   = (xi[-1] - xi[-2])  / max(dt, _EPS)
    return vel


def _fd_acceleration(xi: np.ndarray, dt: float) -> np.ndarray:
    """
    Second derivative via central differences.  Shape: (T, n_dof).
    """
    acc = np.zeros_like(xi)
    acc[1:-1] = (xi[2:] - 2.0 * xi[1:-1] + xi[:-2]) / max(dt ** 2, _EPS)
    acc[0]    = acc[1]
    acc[-1]   = acc[-2]
    return acc


def _fd_jerk(xi: np.ndarray, dt: float) -> np.ndarray:
    """
    Third derivative via central differences.  Shape: (T, n_dof).

    Why jerk instead of torque?
    ---------------------------
    STOMP's original paper uses torque as the smoothness cost because it
    has a physical meaning for manipulators.  Here we substitute jerk
    (third derivative of position) as a proxy — it is model-free, applies
    to any robot type, and matches the _jerk_cost() logic in problem.py.

    Central difference for the third derivative:
        d³q/dt³ ≈ (-q_{t-2} + 2q_{t-1} - 2q_{t+1} + q_{t+2}) / (2 dt³)
    Endpoints are extrapolated from their nearest computed value.
    """
    T   = xi.shape[0]
    jrk = np.zeros_like(xi)
    dt3 = max(dt ** 3, _EPS)

    for t in range(2, T - 2):
        jrk[t] = (-xi[t - 2] + 2.0 * xi[t - 1]
                  - 2.0 * xi[t + 1] + xi[t + 2]) / (2.0 * dt3)

    # Fill boundary stencils with nearest valid value
    if T > 4:
        jrk[0]  = jrk[2]
        jrk[1]  = jrk[2]
        jrk[-1] = jrk[-3]
        jrk[-2] = jrk[-3]

    return jrk


# =============================================================================
# Individual cost terms
# =============================================================================

def _jerk_cost_numpy(
    xi: np.ndarray,
    dt: float,
    duration: float,
    real_indices: list[int],
    virtual_indices: list[int],
    real_weight: float,
    virtual_weight: float,
) -> float:
    """
    STOMP's internal jerk cost.

    This is NOT comparable with problem.py's jerk cost and must not be used
    to compare planners: FOCI sums over samples while this averages over
    waypoints (a factor of T), and a 4-point stencil on a handful of
    waypoints recovers only a fraction of the true jerk of the same curve.
    Use benchmark.utils.jerk_metric for anything cross-planner.

    The duration^6 factor is kept only so that the obstacle and jerk weights
    keep the same order of magnitude across different trajectory lengths.

        cost = weight * duration^6 * mean( jerk² ) over real/virtual joints
    """
    jrk            = _fd_jerk(xi, dt)                # (T, n_dof)
    duration_factor = max(duration, _EPS) ** 6
    cost            = 0.0

    if real_indices:
        real_jerk = jrk[:, real_indices]
        cost += real_weight * duration_factor * float(np.mean(real_jerk ** 2))

    if virtual_indices:
        virtual_jerk = jrk[:, virtual_indices]
        cost += virtual_weight * duration_factor * float(np.mean(virtual_jerk ** 2))

    return cost


def _constraint_violation_cost(
    xi: np.ndarray,
    dt: float,
    joint_groups: JointGroups,
    joint_limits: list[tuple[float, float]] | None,
    n_dof: int,
) -> float:
    """
    Soft penalty for constraint violations.

    Unlike FOCI/CHOMP (which enforce hard bounds via NLP constraints or
    clipping), STOMP adds the *magnitude of the violation* to the cost.
    Noisy samples that violate constraints are thus assigned high cost and
    receive low weight in the update — they are naturally suppressed
    without disrupting the sampling process.

    Three violation types are penalised:

    1. Velocity:
         For real joints:    max(0, |dq/dt| − wmax)
         For virtual joints: max(0, ||dq_virtual/dt||² − virt_wmax²)
         (mirrors the quadratic virtual-joint constraint in problem.py)

    2. Acceleration:
         For real joints:    max(0, |d²q/dt²| − amax)
         For virtual joints: max(0, ||d²q_virtual/dt²||² − virt_amax²)

    3. Joint limits:
         max(0, lower − q)  +  max(0, q − upper)  for each DOF and waypoint.
    """
    vel = _fd_velocity(xi, dt)       # (T, n_dof)
    acc = _fd_acceleration(xi, dt)   # (T, n_dof)
    cost = 0.0

    virtual_set = set(joint_groups.virtual_indices)

    # ---- velocity violations ----
    for t in range(xi.shape[0]):
        virtual_vel_sq = 0.0

        for d in range(n_dof):
            if d in virtual_set:
                virtual_vel_sq += vel[t, d] ** 2
            else:
                viol = max(0.0, abs(vel[t, d]) - joint_groups.real_wmax)
                cost += viol

        if joint_groups.virtual_indices:
            viol = max(0.0, virtual_vel_sq - joint_groups.virtual_wmax ** 2)
            cost += viol

    # ---- acceleration violations ----
    for t in range(xi.shape[0]):
        virtual_acc_sq = 0.0

        for d in range(n_dof):
            if d in virtual_set:
                virtual_acc_sq += acc[t, d] ** 2
            else:
                viol = max(0.0, abs(acc[t, d]) - joint_groups.real_amax)
                cost += viol

        if joint_groups.virtual_indices:
            viol = max(0.0, virtual_acc_sq - joint_groups.virtual_amax ** 2)
            cost += viol

    # ---- joint limit violations ----
    if joint_limits:
        for t in range(xi.shape[0]):
            for d, (lower, upper) in enumerate(joint_limits[:n_dof]):
                lo = -np.inf if lower is None else float(lower)
                hi =  np.inf if upper is None else float(upper)

                if not np.isinf(lo):
                    cost += max(0.0, lo - xi[t, d])
                if not np.isinf(hi):
                    cost += max(0.0, xi[t, d] - hi)

    return cost


# =============================================================================
# Collision cost (CasADi function compiled once)
# =============================================================================

def _build_collision_fn(robot, environment) -> tuple[cas.Function, list]:
    """
    Compile one CasADi function per robot Gaussian:
        collision_fn_g(x: R³) → (cost: R, grad: R³)

    Why per-Gaussian instead of one joint function?
    -----------------------------------------------
    Each Gaussian has its own combined covariance
    (env_cov + robot_cov_g), so the ConvolutionFunctor is
    parameterised differently per Gaussian.  Building one function per
    Gaussian makes the cost evaluation trivially parallelisable and mirrors
    the loop structure of _obstacle_cost() in problem.py exactly.

    Returns
    -------
    fns       : list of K CasADi functions, one per Gaussian
    callbacks : ConvolutionFunctor objects (kept alive to avoid GC)
    """
    robot_covariances = np.asarray(robot.collision_covariances(), dtype=float)
    n_gaussians       = robot_covariances.shape[0]
    fns               = []
    callbacks         = []

    for g in range(n_gaussians):
        x_sym = cas.MX.sym(f"x_stomp_g{g}", 3)
        pt    = x_sym.T                             # (1, 3) — ConvolutionFunctor expects (num_points, 3)

        combined_cov     = environment.obstacle_covariances + robot_covariances[g]
        combined_cov_det = np.linalg.det(combined_cov)
        combined_cov_inv = np.linalg.inv(combined_cov)

        conv = ConvolutionFunctor(
            f"stomp_conv_g{g}",
            1,                                      # num_points = 1 (one waypoint at a time)
            environment.obstacle_means,
            combined_cov_det,
            combined_cov_inv,
        )
        callbacks.append(conv)

        cost = conv(pt)
        # No gradient needed for STOMP: cost is evaluated numerically
        # on sampled trajectories, not differentiated.
        fn   = cas.Function(
            f"stomp_collision_g{g}",
            [x_sym],
            [cost],
            [f"x_g{g}"],
            [f"cost_g{g}"],
        )
        fns.append(fn)

    return fns, callbacks


def _obstacle_cost_trajectory(
    xi: np.ndarray,
    collision_geometry_fn: cas.Function,
    collision_cost_fns: list[cas.Function],
    n_collision_points: int,
) -> float:
    """
    Evaluate the total obstacle cost for a full trajectory xi (T, n_dof).

    For each waypoint t and each robot Gaussian g, we:
      1. Compute the Gaussian's 3-D position via FK (collision_geometry_fn).
      2. Evaluate the convolution cost at that position.
      3. Average over waypoints, Gaussians and obstacles.

    The average over obstacles comes from the functor itself: it is built with
    num_points=1, so its normaliser is 1 / n_obstacles.  What is left is the
    division by n_gaussians AND by T.  Dividing only by n_gaussians -- as this
    did -- leaves a cost T times larger than problem.py's for the same
    geometry, which silently made `weights["obstacle"]` mean something T times
    heavier here than in FOCI and broke the very comparison this planner
    exists for.  With both divisions, weights["obstacle"] is in FOCI's units.
    """
    T    = xi.shape[0]
    cost = 0.0
    n_g  = n_collision_points

    for t in range(T):
        pts, _ = collision_geometry_fn(xi[t])           # (n_g, 3), jacobians (unused)
        pts_np  = np.asarray(pts, dtype=float).reshape(n_g, 3)

        for g in range(n_g):
            c = float(collision_cost_fns[g](pts_np[g]))
            cost += c

    return cost / max(float(n_g) * float(T), _EPS)


# =============================================================================
# STOMP weight computation
# =============================================================================

def _compute_sample_weights(costs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Convert per-sample costs to probability weights using the softmin.

        w_k = exp(-h * (Q_k - min Q) / (max Q - min Q)) / Σ (...)

    Costs are shifted by their minimum AND divided by their range, as in
    Kalakrishnan et al.  The range normalisation is what makes h dimensionless:
    without it, h carries units of 1/cost, and since these costs are dominated
    by a jerk term of order 1e4-1e5, any h around 10 collapses the weights onto
    a single sample and STOMP degenerates into greedy 1-sample random search.

    The temperature h controls how sharply the distribution concentrates on
    the best samples:
      - Large h  → winner-takes-all (only the best sample matters).
      - Small h  → uniform averaging (all samples contribute equally).

    Parameters
    ----------
    costs       : (K,)  per-sample total costs
    temperature : h > 0

    Returns
    -------
    weights : (K,)  normalised, sum to 1
    """
    spread    = float(costs.max() - costs.min())
    shifted   = (costs - costs.min()) / max(spread, _EPS)
    log_w     = -temperature * shifted  # already <= 0, so exp cannot overflow
    weights   = np.exp(log_w)
    total     = weights.sum()
    if total < _EPS:
        weights[:] = 1.0 / len(weights)
    else:
        weights   /= total
    return weights


# =============================================================================
# STOMPPlanner
# =============================================================================

class STOMPPlanner:
    """
    STOMP trajectory optimisation planner.

    Drop-in replacement for ``Planner`` and ``CHOMPPlanner``:

        planner = STOMPPlanner(robot, env, joint_groups, ...)
        result  = planner.plan(start, goal)   # → PlanningResult

    Key difference from CHOMP
    -------------------------
    CHOMP is gradient-based: it differentiates the obstacle cost w.r.t. the
    trajectory and follows the gradient.  STOMP is gradient-free: it samples
    noisy trajectories, scores them, and updates the mean by a
    *probability-weighted average of the noise*.  This makes STOMP suitable
    for cost functions that are non-differentiable (e.g. binary collision
    checks, discrete environment representations).

    Parameters
    ----------
    robot :
        ManipulatorRobot (or any BaseRobot with gaussian_specs,
        collision_covariances(), joint_limits()).
    environment :
        GaussianEnvironment with obstacle_means and obstacle_covariances.
    joint_groups : JointGroups
        Real vs virtual joint definitions and limits.
    num_waypoints : int
        Number of trajectory waypoints T.
    n_samples : int
        K — number of noisy trajectories sampled per iteration.
        More samples → more stable update but higher cost per iteration.
    max_iter : int
        Maximum number of update iterations.
    temperature : float
        h — controls sharpness of the softmin weight distribution.
        Higher values make the update concentrate on the best sample.
    weights : dict[str, float] | None
        Per-term cost weights: 'obstacle', 'jerk', 'constraint'.
    convergence_tol : float
        Stop early when the total cost fails to improve by more than this
        much for `patience` consecutive iterations.  It is measured on the
        cost and not on ||Δξ||, because the step size has a floor set by
        noise_scale and can never reach a small tolerance however good the
        trajectory already is.
    noise_scale : float
        Global scaling applied to all sampled noise perturbations.
        Increase if the optimizer is stuck; decrease for fine-tuning.
    noise_decay : float
        Per-iteration multiplicative decay of noise_scale.  Without it the
        exploration never narrows and the trajectory keeps random-walking
        around the optimum.
    patience : int
        Number of consecutive non-improving iterations tolerated before
        declaring convergence.
    seed : int | None
        Random seed for reproducibility.
    total_time : float
        Normalised trajectory duration used for dt and cost scaling.
        Increase for slower motions, decrease for faster ones.
    """

    def __init__(
        self,
        robot: Any,
        environment: Any,
        joint_groups: JointGroups,
        num_waypoints:   int   = 40,
        n_samples:       int   = 10,
        max_iter:        int   = 200,
        temperature:     float = 10.0,
        weights:         dict[str, float] | None = None,
        convergence_tol: float = 1e-4,
        noise_scale:     float = 0.1,
        noise_decay:     float = 0.99,
        patience:        int   = 20,
        seed:            int | None = None,
        total_time:      float = 1.0,
    ):
        self.robot           = robot
        self.environment     = environment
        self.joint_groups    = joint_groups
        self.num_waypoints   = int(num_waypoints)
        self.n_samples       = int(n_samples)
        self.max_iter        = int(max_iter)
        self.temperature     = float(temperature)
        self.convergence_tol = float(convergence_tol)
        self.noise_scale     = float(noise_scale)
        self.noise_decay     = float(noise_decay)
        self.patience        = int(patience)
        self.total_time      = float(total_time)

        # max_iter >= 1 so the optimisation loop runs at least once: the
        # post-loop cost breakdown reads values the loop body defines.
        # num_waypoints >= 3 so the finite-difference derivatives and the
        # (T+2, T) smoothness operator are well defined.
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}.")
        if self.num_waypoints < 3:
            raise ValueError(
                f"num_waypoints must be at least 3, got {self.num_waypoints}."
            )
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {self.n_samples}.")

        self.weights = {
            "obstacle":   1.0,
            "jerk":       1.0,
            "constraint": 1.0,
            **(weights or {}),
        }

        # dt is the time step between consecutive waypoints.
        # It is used consistently in all derivative estimations (velocity,
        # acceleration, jerk) and in the smoothness matrix, so every cost
        # term has the same time units.
        self.dt = self.total_time / max(self.num_waypoints - 1, 1)

        self._rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Joint information cached once from the robot
        # ------------------------------------------------------------------
        self._joint_limits: list[tuple[float, float]] | None = (
            self.robot.joint_limits()
            if hasattr(self.robot, "joint_limits")
            else None
        )

        robot_covs = np.asarray(self.robot.collision_covariances(), dtype=float)
        if robot_covs.ndim != 3 or robot_covs.shape[1:] != (3, 3):
            raise ValueError(
                "robot.collision_covariances() must have shape (n_gaussians, 3, 3)."
            )
        self._n_collision_points = robot_covs.shape[0]

        # ------------------------------------------------------------------
        # Build pre-compiled CasADi functions (done once in __init__,
        # not inside plan(), to amortise compilation cost over repeated calls)
        # ------------------------------------------------------------------

        # FK + Jacobian for each collision point — needed to map
        # q → 3-D position of each robot Gaussian.
        self._collision_geometry_fn = self._build_collision_geometry_fn()

        # Per-Gaussian convolution cost evaluated at a single 3-D point.
        self._collision_cost_fns, self._callbacks = _build_collision_fn(
            robot, environment
        )

        # Build the smoothness precision matrix R once.
        # R^{-1} (its Cholesky factor L) is used only for noise sampling.
        # It is NOT applied to the update step — see plan() block 3e.
        R           = _build_smoothness_matrix(self.num_waypoints, self.dt)
        self._R     = R
        self._R_inv = _build_precision_inverse(R)   # noise sampling via cholesky(R_inv)
        self._M     = _build_update_matrix(self._R_inv)  # update projection

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_collision_geometry_fn(self) -> cas.Function:
        """
        CasADi function:  q (n_dof,) → points (n_g, 3), J_stack (3*n_g, n_dof)

        Identical to CHOMPPlanner._build_collision_geometry_function().
        Compiled once; evaluated numerically for every (sample, waypoint).
        """
        q   = cas.MX.sym("q_stomp_geom", self.robot.n_dof)
        pts = self.robot.collision_points(q)             # (n_g, 3)

        jacobians = []
        for g in range(self._n_collision_points):
            pt_g = pts[g, :].T
            jacobians.append(cas.jacobian(pt_g, q))

        J_stack = cas.vertcat(*jacobians)                # (3*n_g, n_dof)

        return cas.Function(
            "stomp_collision_geometry",
            [q],
            [pts, J_stack],
        )

    def _total_cost(self, xi: np.ndarray) -> float:
        """
        Evaluate the total scalar cost of a trajectory xi (T, n_dof).

        Three additive terms — all weighted by self.weights:

        1. Obstacle cost
           Gaussian convolution summed over waypoints and body points,
           normalised by n_gaussians (mirrors _obstacle_cost() in problem.py).

        2. Jerk cost
           Matches _jerk_cost() in problem.py: scales by duration^6 and
           averages over real/virtual joints separately.

        3. Constraint violation cost
           Sum of magnitudes of velocity, acceleration, and joint-limit
           violations.  This replaces the hard NLP constraints of problem.py
           with soft penalties appropriate for a sampling-based method.
        """
        n_dof        = self.robot.n_dof
        real_indices = self.joint_groups.real_indices(n_dof)
        virtual_idx  = self.joint_groups.virtual_indices

        obs_cost = _obstacle_cost_trajectory(
            xi,
            self._collision_geometry_fn,
            self._collision_cost_fns,
            self._n_collision_points,
        )

        jrk_cost = _jerk_cost_numpy(
            xi,
            dt              = self.dt,
            duration        = self.total_time,
            real_indices    = real_indices,
            virtual_indices = virtual_idx,
            real_weight     = 1.0,    # outer weight applied below
            virtual_weight  = 1.0,
        )

        con_cost = _constraint_violation_cost(
            xi,
            dt            = self.dt,
            joint_groups  = self.joint_groups,
            joint_limits  = self._joint_limits,
            n_dof         = n_dof,
        )

        return (
            self.weights["obstacle"]   * obs_cost
            + self.weights["jerk"]     * jrk_cost
            + self.weights["constraint"] * con_cost
        )

    def _straight_line(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        s = np.linspace(0.0, 1.0, self.num_waypoints)[:, None]
        return (1.0 - s) * start[None, :] + s * goal[None, :]

    def _clip_to_joint_limits(self, xi: np.ndarray) -> np.ndarray:
        """
        Hard-clip a trajectory to the robot's joint limits.

        In STOMP this is applied only to the *mean* trajectory after the
        update step — not to the noisy samples.  Clipping the samples would
        bias the noise distribution and corrupt the probability weights.
        The joint-limit *cost* on the samples already discourages violations
        without distorting the distribution.
        """
        if not self._joint_limits:
            return xi

        out = xi.copy()
        for d, (lo, hi) in enumerate(self._joint_limits[: xi.shape[1]]):
            lo_f = -np.inf if lo is None else float(lo)
            hi_f =  np.inf if hi is None else float(hi)
            out[:, d] = np.clip(out[:, d], lo_f, hi_f)
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: np.ndarray,
        goal:  np.ndarray,
        initial_trajectory: np.ndarray | None = None,
    ) -> PlanningResult:
        """
        Run STOMP optimisation and return a PlanningResult.

        Parameters
        ----------
        start : (n_dof,)
            Initial joint configuration.
        goal : (n_dof,)
            Goal joint configuration.
        initial_trajectory : (N, n_dof) | None
            Optional warm-start trajectory (e.g. from FOCI).
            If None, a straight-line initialisation is used.

        Returns
        -------
        PlanningResult
            Same structure as Planner.plan() and CHOMPPlanner.plan().
        """
        total_start = perf_counter()

        start = np.asarray(start, dtype=float).reshape(-1)
        goal  = np.asarray(goal,  dtype=float).reshape(-1)
        n_dof = len(start)

        self.joint_groups.validate(n_dof)

        if goal.shape != start.shape:
            raise ValueError(
                f"goal must have shape {start.shape}, got {goal.shape}."
            )

        # ==============================================================
        # Block 1 – Initialiser
        # ==============================================================
        init_start = perf_counter()

        if initial_trajectory is None:
            # Straight-line: minimum-smoothness boundary-satisfying trajectory.
            xi = self._straight_line(start, goal)
        else:
            xi = np.asarray(initial_trajectory, dtype=float)
            if xi.ndim != 2 or xi.shape[1] != n_dof:
                raise ValueError(
                    f"initial_trajectory must have shape (N, {n_dof}), got {xi.shape}."
                )
            # Resample to num_waypoints if needed
            if xi.shape[0] != self.num_waypoints:
                old_s = np.linspace(0.0, 1.0, xi.shape[0])
                new_s = np.linspace(0.0, 1.0, self.num_waypoints)
                xi    = np.vstack(
                    [np.interp(new_s, old_s, xi[:, d]) for d in range(n_dof)]
                ).T

        xi[0]  = start
        xi[-1] = goal
        initial_xi = xi.copy()

        init_time = perf_counter() - init_start

        # ==============================================================
        # Block 2 – Build  (matrices already prepared in __init__)
        # ==============================================================
        build_start = perf_counter()
        # R and R_inv were built in __init__ to amortise compilation.
        # We just record the timing for consistency with Planner.
        build_time = perf_counter() - build_start

        # ==============================================================
        # Block 3 – STOMP optimisation loop
        # ==============================================================
        solve_start = perf_counter()

        converged              = False
        best_cost              = np.inf
        stalled_iters          = 0
        iterations_run         = 0
        final_cost             = np.inf
        final_obstacle_cost    = np.inf
        final_jerk_cost        = np.inf
        final_constraint_cost  = np.inf
        final_update_norm      = np.inf
        # Defined before the loop: the post-loop breakdown reads it, and it
        # would be an unbound local if the loop body never ran.
        update_norm            = np.inf

        for iteration in range(self.max_iter):
            iterations_run = iteration + 1

            # ----------------------------------------------------------
            # 3a. Sample K noisy trajectories
            # ----------------------------------------------------------
            # noise shape: (K, T, n_dof)
            # Each noise[k] is drawn from N(0, R^{-1}) so it is smooth
            # by construction and has zero mean across samples.
            decay = self.noise_decay ** iteration
            noise = (
                self.noise_scale
                * decay
                * _sample_noise(self._R_inv, n_dof, self.n_samples, self._rng)
            )

            # ----------------------------------------------------------
            # 3b. Evaluate cost of each noisy trajectory
            # ----------------------------------------------------------
            # xi_k = xi + noise[k], with endpoints re-pinned.
            # Costs are evaluated on the noisy trajectory, not on xi.
            costs = np.zeros(self.n_samples, dtype=float)

            for k in range(self.n_samples):
                xi_k          = xi + noise[k]
                xi_k[0]       = start
                xi_k[-1]      = goal
                costs[k]      = self._total_cost(xi_k)

            # ----------------------------------------------------------
            # 3c. Compute sample weights  w_k ∝ exp(-h · Q_k)
            # ----------------------------------------------------------
            weights = _compute_sample_weights(costs, self.temperature)  # (K,)

            # ----------------------------------------------------------
            # 3d. Compute the probability-weighted noise sum
            # ----------------------------------------------------------
            # delta_xi[t, d] = Σ_k w_k * noise[k, t, d]
            # Shape: (T, n_dof)
            delta_xi = np.einsum("k,ktd->td", weights, noise)           # (T, n_dof)

            # ----------------------------------------------------------
            # 3e. Apply the update  Δξ = M · Σ_k w_k δξ_k
            # ----------------------------------------------------------
            # M is the paper's update projection (see _build_update_matrix).
            # The probability-weighted noise sum is not smooth by itself --
            # the weights depend on the costs -- so M is what keeps the
            # trajectory smooth.  Its column normalisation bounds the step,
            # which is what an earlier version was missing when it dropped
            # the projection to stop the update exploding.
            xi_new = xi + self._M @ delta_xi

            # ----------------------------------------------------------
            # 3f. Re-pin endpoints and clip to joint limits
            # ----------------------------------------------------------
            # Clipping is applied only to the mean trajectory (not to
            # the noisy samples) to preserve the noise distribution.
            xi_new        = self._clip_to_joint_limits(xi_new)
            xi_new[0]     = start
            xi_new[-1]    = goal

            # ----------------------------------------------------------
            # 3g. Convergence check
            # ----------------------------------------------------------
            update_norm = float(np.linalg.norm(xi_new - xi))
            xi          = xi_new

            # Convergence on the cost, not on the step: ||Δξ|| is bounded
            # below by the sampled noise, so testing it against a tolerance
            # like 1e-3 can never succeed.
            current_cost = self._total_cost(xi)

            if best_cost - current_cost < self.convergence_tol:
                stalled_iters += 1
                if stalled_iters >= self.patience:
                    converged = True
                    break
            else:
                best_cost     = current_cost
                stalled_iters = 0

        # ----------------------------------------------------------
        # Final cost breakdown on the converged mean trajectory
        # ----------------------------------------------------------
        n_dof_final   = self.robot.n_dof
        real_indices  = self.joint_groups.real_indices(n_dof_final)
        virtual_idx   = self.joint_groups.virtual_indices

        final_obstacle_cost = self.weights["obstacle"] * _obstacle_cost_trajectory(
            xi,
            self._collision_geometry_fn,
            self._collision_cost_fns,
            self._n_collision_points,
        )
        final_jerk_cost = self.weights["jerk"] * _jerk_cost_numpy(
            xi,
            dt              = self.dt,
            duration        = self.total_time,
            real_indices    = real_indices,
            virtual_indices = virtual_idx,
            real_weight     = 1.0,
            virtual_weight  = 1.0,
        )
        final_constraint_cost = self.weights["constraint"] * _constraint_violation_cost(
            xi,
            dt           = self.dt,
            joint_groups = self.joint_groups,
            joint_limits = self._joint_limits,
            n_dof        = n_dof_final,
        )
        final_cost     = final_obstacle_cost + final_jerk_cost + final_constraint_cost
        final_update_norm = update_norm

        # The constraint term above is a soft penalty whose weight relative to
        # the jerk term is arbitrary, so the returned trajectory may well
        # violate the limits.  Report the uniform time scaling that would make
        # it feasible, which is comparable across planners.
        # num_samples=self.num_waypoints, NOT the default 200: the metric is
        # resolution dependent (see src/benchmark/utils.py) and leaving the
        # default inflates the scale by ~3x on a 12-waypoint trajectory, so
        # this metadata would contradict the benchmark table, which passes
        # num_waypoints.
        limit_scale, feasible_duration = limit_scaling_factor(
            xi,
            duration=self.total_time,
            joint_groups=self.joint_groups,
            n_dof=n_dof_final,
            num_samples=self.num_waypoints,
        )

        solve_time = perf_counter() - solve_start
        total_time = perf_counter() - total_start

        # ==============================================================
        # Block 4 – Pack PlanningResult
        # ==============================================================
        return PlanningResult(
            trajectory         = xi,
            control_points     = None,
            initial_trajectory = initial_xi,
            success            = converged,
            timings={
                "initializer": init_time,
                "build":       build_time,
                "solve":       solve_time,
                "total":       total_time,
            },
            solver_stats={},
            metadata={
                "iterations":            iterations_run,
                "final_cost":            final_cost,
                "final_obstacle_cost":   final_obstacle_cost,
                "final_jerk_cost":       final_jerk_cost,
                "final_constraint_cost": final_constraint_cost,
                "final_update_norm":     final_update_norm,
                "converged":             converged,
                "num_waypoints":         self.num_waypoints,
                "n_samples":             self.n_samples,
                "temperature":           self.temperature,
                "noise_scale":           self.noise_scale,
                "noise_decay":           self.noise_decay,
                "patience":              self.patience,
                "weights":               self.weights,
                "dt":                    self.dt,
                "total_time":            self.total_time,
                "limit_scale":           limit_scale,
                "feasible_duration":     feasible_duration,
            },
        )