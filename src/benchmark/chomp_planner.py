"""
Basic CHOMP planner with fixed start and fixed final configuration.

This version follows the basic CHOMP setup:
- The trajectory is optimized in configuration space.
- The start configuration q0 and final configuration q1 are fixed.
- Only interior waypoints are updated.
- The objective is obstacle cost + lambda * smoothness cost.
- The update uses the CHOMP metric A, i.e. a covariant step A^{-1} grad.

The task-space goal conversion is intentionally not handled here. If your
high-level planner produces a final configuration, pass that configuration as
`goal`.
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


def _straight_line_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    num_waypoints: int,
) -> np.ndarray:
    s = np.linspace(0.0, 1.0, num_waypoints)[:, None]
    return (1.0 - s) * start[None, :] + s * goal[None, :]


def _build_chomp_metric_matrix(
    num_waypoints: int,
    dt: float,
    ridge: float = 1e-8,
) -> np.ndarray:
    """
    Build the finite-dimensional CHOMP metric A for the interior waypoints.

    The full trajectory has T waypoints, but the endpoints are fixed. Therefore
    the optimization variable contains only T - 2 interior waypoints.

    For the basic squared-velocity smoothness functional,

        F_smooth = 0.5 * integral ||dq/dt||^2 dt,

    the finite-difference gradient on interior waypoints is proportional to

        2 q_t - q_{t-1} - q_{t+1}.

    Hence A is the tridiagonal matrix with 2 on the diagonal and -1 on the
    first off-diagonals, scaled by 1 / dt.
    """
    if num_waypoints < 3:
        raise ValueError("num_waypoints must be at least 3.")

    num_internal = num_waypoints - 2
    A = 2.0 * np.eye(num_internal, dtype=float)

    if num_internal > 1:
        off_diag = -1.0 * np.ones(num_internal - 1, dtype=float)
        A += np.diag(off_diag, k=1)
        A += np.diag(off_diag, k=-1)

    A /= max(float(dt), _EPS)
    A += ridge * np.eye(num_internal, dtype=float)
    return A


def _smoothness_gradient_full(xi: np.ndarray, dt: float) -> np.ndarray:
    """
    Gradient of the squared-velocity smoothness cost w.r.t. all waypoints.

    Endpoints are filled with zero because they are fixed and are not optimized.
    """
    grad = np.zeros_like(xi)
    grad[1:-1] = (2.0 * xi[1:-1] - xi[:-2] - xi[2:]) / max(float(dt), _EPS)
    return grad


def _smoothness_cost(xi: np.ndarray, dt: float) -> float:
    diff = np.diff(xi, axis=0)
    return 0.5 * float(np.sum(diff * diff)) / max(float(dt), _EPS)


class CHOMPPlanner:
    def __init__(
        self,
        robot: Any,
        environment: Any,
        joint_groups: JointGroups,
        num_waypoints: int = 40,
        max_iter: int = 200,
        learning_rate: float = 0.01,
        weights: dict[str, float] | None = None,
        convergence_tol: float = 1e-4,
        joint_limit_margin: float = 1e-6,
        total_time: float = 1.0,
    ):
        self.robot = robot
        self.environment = environment
        self.joint_groups = joint_groups
        self.num_waypoints = int(num_waypoints)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.convergence_tol = float(convergence_tol)
        self.joint_limit_margin = float(joint_limit_margin)

        self.weights = {
            "obstacle": 1.0,
            "smoothness": 1.0,
            **(weights or {}),
        }

        # Horizon of the trajectory.  It must be settable: the covariant step
        # balances the obstacle gradient (dt-invariant) against the smoothness
        # gradient (scales as 1/dt), so a hardcoded 1.0 makes the same nominal
        # weights mean something 2-5x different from STOMP's, which is given
        # total_time=2.0 or 5.0 by the benchmarks.
        self.total_time = float(total_time)
        self.dt = self.total_time / max(self.num_waypoints - 1, 1)

        self._robot_covariances = self._load_robot_collision_covariances()
        self._n_collision_points = int(self._robot_covariances.shape[0])

        self._callbacks: list[Any] = []
        self._collision_geometry_fun = self._build_collision_geometry_function()
        self._workspace_cost_grad_funs = self._build_workspace_cost_gradient_functions()
        self._ee_fun = self._build_ee_function()

    def _load_robot_collision_covariances(self) -> np.ndarray:
        robot_covariances = np.asarray(self.robot.collision_covariances(), dtype=float)

        if robot_covariances.ndim != 3 or robot_covariances.shape[1:] != (3, 3):
            raise ValueError(
                "robot.collision_covariances() must have shape "
                "(n_collision_points, 3, 3)."
            )

        if robot_covariances.shape[0] == 0:
            raise ValueError("robot must define at least one collision Gaussian.")

        return robot_covariances

    def _build_ee_function(self) -> cas.Function:
        q = cas.MX.sym("q_ee", self.robot.n_dof)
        return cas.Function("chomp_ee", [q], [self.robot.f_task(q)])

    def _build_collision_geometry_function(self) -> cas.Function:
        """
        Return body-point positions and their kinematic Jacobians.

        Output:
            points:  shape (n_collision_points, 3)
            J_stack: shape (3 * n_collision_points, n_dof)
                     rows 3*i:3*i+3 are the Jacobian of point i.
        """
        q = cas.MX.sym("q_chomp_geom", self.robot.n_dof)
        points = self.robot.collision_points(q)

        jacobians = []
        for point_idx in range(self._n_collision_points):
            point_i = points[point_idx, :].T
            jacobians.append(cas.jacobian(point_i, q))

        J_stack = cas.vertcat(*jacobians)

        return cas.Function(
            "chomp_collision_geometry",
            [q],
            [points, J_stack],
        )

    def _build_workspace_cost_gradient_functions(self) -> list[cas.Function]:
        """
        Build one workspace cost/gradient function per robot body point.

        The cost for each robot Gaussian uses the convolution between the
        environment Gaussian obstacles and that robot Gaussian.
        """
        funs: list[cas.Function] = []

        for point_idx in range(self._n_collision_points):
            x = cas.MX.sym(f"x_chomp_obs_{point_idx}", 3)
            point = x.T  # ConvolutionFunctor expects shape (num_points, 3).

            covs = self.environment.obstacle_covariances + self._robot_covariances[point_idx]
            covs_det = np.linalg.det(covs)
            covs_inv = np.linalg.inv(covs)

            convolution = ConvolutionFunctor(
                f"chomp_workspace_obs_{point_idx}",
                1,                       # num_points = 1 (one waypoint at a time)
                self.environment.obstacle_means,
                covs_det,
                covs_inv,
            )

            self._callbacks.append(convolution)

            cost = convolution(point)
            grad = cas.gradient(cost, x)

            funs.append(
                cas.Function(
                    f"chomp_workspace_cost_grad_{point_idx}",
                    [x],
                    [cost, grad],
                )
            )

        return funs

    def _collision_geometry(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points, J_stack = self._collision_geometry_fun(q)

        points_np = np.asarray(points, dtype=float).reshape(self._n_collision_points, 3)
        J_np = np.asarray(J_stack, dtype=float).reshape(
            self._n_collision_points,
            3,
            self.robot.n_dof,
        )

        return points_np, J_np

    def _workspace_cost_and_grad(
        self,
        point_idx: int,
        x: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        cost, grad = self._workspace_cost_grad_funs[point_idx](np.asarray(x, dtype=float).reshape(3))
        return float(cost), np.asarray(grad, dtype=float).reshape(3)

    def _ee_position(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self._ee_fun(q), dtype=float).reshape(-1)

    def _resample_trajectory(self, xi: np.ndarray) -> np.ndarray:
        if xi.ndim != 2:
            raise ValueError("initial_trajectory must be a 2D array with shape (N, n_dof).")

        if xi.shape[0] < 2:
            raise ValueError("initial_trajectory must contain at least two waypoints.")

        if xi.shape[0] == self.num_waypoints:
            return xi.copy()

        old_s = np.linspace(0.0, 1.0, xi.shape[0])
        new_s = np.linspace(0.0, 1.0, self.num_waypoints)

        return np.vstack(
            [np.interp(new_s, old_s, xi[:, dof]) for dof in range(xi.shape[1])]
        ).T

    def _clip_to_joint_limits(self, xi: np.ndarray) -> np.ndarray:
        if not hasattr(self.robot, "joint_limits"):
            return xi

        limits = self.robot.joint_limits()
        if not limits:
            return xi

        out = xi.copy()

        for dof, (lower, upper) in enumerate(limits[: xi.shape[1]]):
            lo = -np.inf if lower is None else float(lower) + self.joint_limit_margin
            hi = np.inf if upper is None else float(upper) - self.joint_limit_margin
            out[:, dof] = np.clip(out[:, dof], lo, hi)

        return out

    def _initialize_trajectory(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        initial_trajectory: np.ndarray | None,
    ) -> np.ndarray:
        if initial_trajectory is None:
            xi = _straight_line_trajectory(start, goal, self.num_waypoints)
        else:
            xi = np.asarray(initial_trajectory, dtype=float)

            if xi.ndim != 2 or xi.shape[1] != len(start):
                raise ValueError(
                    f"initial_trajectory must have shape (N, {len(start)}). "
                    f"Got {xi.shape}."
                )

            xi = self._resample_trajectory(xi)

        # The endpoints are hard constraints.
        xi[0] = start
        xi[-1] = goal

        xi = self._clip_to_joint_limits(xi)

        # Re-impose endpoints after clipping. If the external planner gives a
        # valid final configuration, this preserves it exactly.
        xi[0] = start
        xi[-1] = goal

        return xi

    def _obstacle_cost_and_gradient_full(self, xi: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Approximate the CHOMP obstacle functional gradient.

        For each robot body point x(q, u), CHOMP uses the arc-length weighted
        workspace obstacle functional and maps its workspace gradient back to
        configuration space through the body-point Jacobian:

            J^T ||x_dot|| [ (I - x_hat x_hat^T) grad c - c kappa ].

        The endpoints are not assigned obstacle gradients because they are fixed.
        """
        T = xi.shape[0]
        grad = np.zeros_like(xi)
        obstacle_cost = 0.0

        points = np.zeros((T, self._n_collision_points, 3), dtype=float)
        jacobians = np.zeros((T, self._n_collision_points, 3, self.robot.n_dof), dtype=float)

        for t in range(T):
            points[t], jacobians[t] = self._collision_geometry(xi[t])

        eye3 = np.eye(3, dtype=float)
        inv_n_points = 1.0 / float(self._n_collision_points)

        for t in range(1, T - 1):
            x_dot = (points[t + 1] - points[t - 1]) / (2.0 * max(self.dt, _EPS))
            x_ddot = (points[t + 1] - 2.0 * points[t] + points[t - 1]) / (
                max(self.dt, _EPS) ** 2
            )

            for point_idx in range(self._n_collision_points):
                cost, cost_grad = self._workspace_cost_and_grad(
                    point_idx,
                    points[t, point_idx],
                )

                velocity = x_dot[point_idx]
                acceleration = x_ddot[point_idx]
                speed = float(np.linalg.norm(velocity))

                if speed > _EPS:
                    tangent = velocity / speed
                    projection = eye3 - np.outer(tangent, tangent)
                    curvature = (projection @ acceleration) / max(speed * speed, _EPS)
                else:
                    # Degenerate local motion: the orthogonal direction is
                    # undefined. Use the full workspace gradient and zero
                    # curvature as a stable fallback.
                    projection = eye3
                    curvature = np.zeros(3, dtype=float)
                    speed = _EPS

                workspace_grad = speed * (projection @ cost_grad - cost * curvature)

                J = jacobians[t, point_idx]
                grad[t] += inv_n_points * (J.T @ workspace_grad) * self.dt
                obstacle_cost += inv_n_points * cost * speed * self.dt

        return obstacle_cost, grad

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        initial_trajectory: np.ndarray | None = None,
    ) -> PlanningResult:
        total_start = perf_counter()

        start = np.asarray(start, dtype=float).reshape(-1)
        goal = np.asarray(goal, dtype=float).reshape(-1)

        self.joint_groups.validate(len(start))

        init_start = perf_counter()
        xi = self._initialize_trajectory(start, goal, initial_trajectory)
        initial_xi = xi.copy()
        init_time = perf_counter() - init_start

        build_start = perf_counter()
        A = _build_chomp_metric_matrix(self.num_waypoints, dt=self.dt)
        build_time = perf_counter() - build_start

        solve_start = perf_counter()

        converged = False
        iterations_run = 0
        final_cost = np.inf
        final_obstacle_cost = np.inf
        final_smoothness_cost = np.inf
        final_update_norm = np.inf
        final_gradient_norm = np.inf

        for iteration in range(self.max_iter):
            iterations_run = iteration + 1

            smooth_grad = _smoothness_gradient_full(xi, dt=self.dt)
            obstacle_cost_raw, obstacle_grad = self._obstacle_cost_and_gradient_full(xi)

            grad = (
                self.weights["smoothness"] * smooth_grad
                + self.weights["obstacle"] * obstacle_grad
            )

            # Fixed endpoint constraints: only interior waypoints are optimized.
            grad_inner = grad[1:-1]

            # Covariant CHOMP step: xi <- xi - alpha * A^{-1} grad.
            # np.linalg.solve avoids forming A^{-1} explicitly.
            covariant_step = np.linalg.solve(A, grad_inner)

            xi_new = xi.copy()
            xi_new[1:-1] -= self.learning_rate * covariant_step

            xi_new = self._clip_to_joint_limits(xi_new)

            # Hard constraints: endpoints stay exactly fixed.
            xi_new[0] = start
            xi_new[-1] = goal

            update_norm = float(np.linalg.norm(xi_new - xi))
            gradient_norm = float(np.linalg.norm(grad_inner))

            xi = xi_new

            final_obstacle_cost = self.weights["obstacle"] * obstacle_cost_raw
            final_smoothness_cost = self.weights["smoothness"] * _smoothness_cost(xi, dt=self.dt)
            final_cost = final_obstacle_cost + final_smoothness_cost
            final_update_norm = update_norm
            final_gradient_norm = gradient_norm

            if update_norm < self.convergence_tol:
                converged = True
                break

        solve_time = perf_counter() - solve_start
        total_time = perf_counter() - total_start

        final_ee = self._ee_position(xi[-1])
        goal_error = float(np.linalg.norm(xi[-1] - goal))

        # CHOMP's objective has no velocity/acceleration term at all, so the
        # trajectory it returns is not dynamically feasible in general.
        # Report the uniform time scaling that would make it feasible, so it
        # can be compared against a planner that enforces the limits.
        # num_samples=self.num_waypoints, NOT the default 200: the metric is
        # resolution dependent (see src/benchmark/utils.py) and leaving the
        # default inflates the scale by ~3x on a 12-waypoint trajectory, so
        # this metadata would contradict the benchmark table, which passes
        # num_waypoints.
        limit_scale, feasible_duration = limit_scaling_factor(
            xi,
            duration=self.total_time,
            joint_groups=self.joint_groups,
            n_dof=int(xi.shape[1]),
            num_samples=self.num_waypoints,
        )

        return PlanningResult(
            trajectory=xi,
            control_points=None,
            initial_trajectory=initial_xi,
            success=converged,
            timings={
                "initializer": init_time,
                "build": build_time,
                "solve": solve_time,
                "total": total_time,
            },
            solver_stats={},
            metadata={
                "iterations": iterations_run,
                "final_cost": final_cost,
                "final_obstacle_cost": final_obstacle_cost,
                "final_smoothness_cost": final_smoothness_cost,
                "final_goal_error_configuration": goal_error,
                "final_ee": final_ee,
                "final_update_norm": final_update_norm,
                "final_gradient_norm": final_gradient_norm,
                "converged": converged,
                "num_waypoints": self.num_waypoints,
                "learning_rate": self.learning_rate,
                "weights": self.weights,
                "dt": self.dt,
                "total_time": self.total_time,
                "limit_scale": limit_scale,
                "feasible_duration": feasible_duration,
                "endpoint_mode": "fixed_start_and_fixed_configuration_goal",
                "optimized_waypoints": "interior_only",
            },
        )