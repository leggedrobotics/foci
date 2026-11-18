"""Abstract base class for trajectory optimisation planners.

Both the static (Planner) and online (OnlinePlanner) variants share the same
__init__ and plan() logic.  The only concrete difference is which
build_problem function is called, so subclasses implement _build_problem().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np

from src.initialize.uniform_initializer import UniformControlPointsInitializer
from src.optimization.solver import create_solver
from src.planning.joints import JointGroups
from src.planning.result import PlanningResult
from src.splines.bspline import BSpline


class BasePlanner(ABC):
    """Abstract trajectory optimisation planner.

    Works with any robot implementing the BaseRobot API and any environment
    implementing BaseGaussianEnvironment.

    Parameters
    ----------
    robot:
        Robot model (e.g. ManipulatorRobot, DroneRobot).
    environment:
        Obstacle environment.
    joint_groups:
        Grouping of joints used when building the optimisation problem.
    initializer:
        Strategy for producing an initial guess. Defaults to
        UniformControlPointsInitializer when None.
    num_control_points:
        Number of B-spline control points.
    num_samples:
        Number of trajectory samples evaluated during optimisation.
    weights:
        Cost-term weights forwarded to build_problem.
    vmax:
        Maximum joint velocity.
    linear_solver:
        IPOPT linear solver backend (e.g. "mumps", "ma27").
    """

    def __init__(
        self,
        robot,
        environment,
        joint_groups: JointGroups,
        initializer=None,
        num_control_points: int = 8,
        num_samples: int = 30,
        weights: dict[str, float] | None = None,
        vmax: float = 1.0,
        linear_solver: str = "mumps",
    ) -> None:
        self.robot = robot
        self.environment = environment
        self.joint_groups = joint_groups
        self.initializer = (
            initializer if initializer is not None else UniformControlPointsInitializer()
        )
        self.num_control_points = num_control_points
        self.num_samples = num_samples
        self.weights = weights
        self.vmax = vmax
        self.linear_solver = linear_solver

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_problem(self) -> tuple:
        """Call the appropriate build_problem function and return
        (nlp, lbg, ubg, callbacks).
        """

    # Subclasses whose NLP structure never depends on call-time state can
    # set this to True to build the NLP and IPOPT solver ONCE (on the first
    # plan() call) and reuse them on every subsequent call instead of
    # rebuilding from scratch each time -- start/goal only ever enter
    # through the parameter vector `p` and the initial guess `x0`, both
    # resolved at solve time, so the symbolic problem itself is identical
    # across calls and reusing it changes nothing about the result.
    #
    # OnlinePlanner must NOT opt in: its _build_problem() re-queries
    # environment.obstacle_means_at(k) and bakes those obstacle positions
    # directly into the symbolic graph rather than passing them as a
    # parameter, so a stale cached build would silently keep using
    # wherever the obstacles were on the first call.
    _cache_built_problem: bool = False

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _variable_bounds(self):
        """
        Joint limits as per-variable bounds on the decision vector.

        `x` is vec(control_points) with control_points of shape
        (num_control_points, n_dof), and CasADi's vec() is column-major, so
        the num_control_points variables belonging to one joint are
        contiguous: that is np.repeat, not np.tile.

        Bounding the control points is conservative but sound: by the convex
        hull property of a B-spline the curve lies inside the convex hull of
        its control points, so control points within the limits imply a
        trajectory within the limits.

        Returns (None, None) when the robot does not declare limits.
        """
        get_limits = getattr(self.robot, "joint_limits", None)
        limits = get_limits() if callable(get_limits) else None

        if limits is None:
            return None, None

        limits = list(limits)
        n_dof = self.robot.n_dof

        if len(limits) != n_dof:
            raise ValueError(
                f"{type(self.robot).__name__}.joint_limits() returned "
                f"{len(limits)} entries for {n_dof} degrees of freedom; the "
                "bounds would be assigned to the wrong joints."
            )

        lower = np.repeat(
            np.array([float(lo) for lo, _ in limits]), self.num_control_points
        )
        upper = np.repeat(
            np.array([float(hi) for _, hi in limits]), self.num_control_points
        )

        return lower, upper

    def _build_parameter_vector(
        self, start: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """Concatenate start and goal into the NLP parameter vector."""
        return np.concatenate([
            np.asarray(start, dtype=float).reshape(-1),
            np.asarray(goal, dtype=float).reshape(-1),
        ])

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def plan(self, start: np.ndarray, goal: np.ndarray) -> PlanningResult:
        """Run trajectory optimisation from start to goal.

        Parameters
        ----------
        start:
            Initial robot configuration.
        goal:
            Goal task-space position.

        Returns
        -------
        PlanningResult
        """
        total_start = perf_counter()

        # --- Initialisation ------------------------------------------
        init_start = perf_counter()

        init_result = self.initializer.initialize(
            robot=self.robot,
            environment=self.environment,
            start=start,
            goal=goal,
            num_control_points=self.num_control_points,
        )

        initial_control_points = init_result.control_points
        initial_trajectory = init_result.trajectory
        init_time = perf_counter() - init_start

        # --- Build optimisation problem -------------------------------
        build_start = perf_counter()

        cached = getattr(self, "_cached_problem", None)

        if self._cache_built_problem and cached is not None:
            nlp, lbg, ubg, callbacks, solver = cached
        else:
            nlp, lbg, ubg, callbacks = self._build_problem()
            solver = create_solver(
                nlp, options={"ipopt.linear_solver": self.linear_solver}
            )

            if self._cache_built_problem:
                self._cached_problem = (nlp, lbg, ubg, callbacks, solver)

        self._casadi_callbacks = callbacks

        build_time = perf_counter() - build_start

        # --- Initial guess -------------------------------------------
        x0 = (
            np.tile(start, (self.num_control_points, 1))
            if initial_control_points is None
            else initial_control_points
        )
        x0 = np.asarray(x0, dtype=float).reshape(-1, order="F")

        lbx, ubx = self._variable_bounds()

        if lbx is not None:
            # The initializer knows nothing about the joint limits -- the
            # RRT* least-squares fit in particular can overshoot past the
            # extremes of the path it is fitting -- so an unclipped guess can
            # start outside its own bounds.  IPOPT would silently drag it
            # inside; clipping here keeps the guess we actually chose.
            x0 = np.clip(x0, lbx, ubx)

        # --- Solve NLP -----------------------------------------------
        solve_start = perf_counter()

        solver_arguments = {
            "x0": x0,
            "p": self._build_parameter_vector(start, goal),
            "lbg": lbg,
            "ubg": ubg,
        }

        if lbx is not None:
            solver_arguments["lbx"] = lbx
            solver_arguments["ubx"] = ubx

        solution = solver(**solver_arguments)

        solve_time = perf_counter() - solve_start

        # --- Extract solution ----------------------------------------
        control_points = np.array(solution["x"]).reshape(
            self.num_control_points, self.robot.n_dof, order="F"
        )

        trajectory = np.array(BSpline(control_points).spline_eval(self.num_samples))

        total_time = perf_counter() - total_start

        # --- Return result -------------------------------------------
        stats = solver.stats()

        return PlanningResult(
            trajectory=trajectory,
            control_points=control_points,
            initial_trajectory=initial_trajectory,
            success=stats.get("success", False),
            timings={
                "initializer": init_time,
                "build": build_time,
                "solve": solve_time,
                "total": total_time,
            },
            solver_stats=stats,
        )