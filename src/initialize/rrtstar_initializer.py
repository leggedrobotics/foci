from __future__ import annotations


import warnings

from time import perf_counter
from typing import Optional

import numpy as np
import open3d as o3d

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from src.splines.bspline import BSpline
from src.initialize.initializer import InitializerResult, PathInitializer


# How many path samples per control point are used to fit the initial guess.
# The fit must be over-determined, otherwise the spline interpolates the path
# and can overshoot around corners.
_PATH_SAMPLES_PER_CONTROL_POINT: int = 5

# Wall-clock budget used when `max_time` is None.  The search MUST be bounded:
# whenever RRT* stores no solution at all -- an invalid start state (a start
# configuration in collision) reports "Invalid start" on every call -- the old
# `while True` with max_time=None spun forever with no diagnostic.  Measured
# with ompl 1.7.0.
_DEFAULT_MAX_TIME: float = 10.0

# Half-width of the artificial box substituted for an infinite joint limit,
# centred on the start configuration.  RRT* samples its state space uniformly,
# so an enormous box (the old +-1e6 clamp gave a 2e6-wide one) destroys the
# sampling density and the tree never gets anywhere near the goal.
_DEFAULT_UNBOUNDED_RADIUS: float = 50.0

# RRT* keeps the tree node closest to the goal as its approximate solution and
# replaces it only when it finds a better one.  Once that distance stops
# improving the search has plateaued and sitting out the rest of the budget
# only burns wall time: the returned path will not change.  Waiting out the
# full budget on a goal_threshold that is effectively unreachable cost ~10 s on
# every call, where the previous code returned this same path after one slice
# (mislabelled as an exact success).  Plateauing has to be just as cheap.
_DEFAULT_APPROXIMATE_PATIENCE: int = 10
_DEFAULT_IMPROVEMENT_TOL: float = 1e-4


def fit_control_points(
    path_points: np.ndarray,
    num_control_points: int,
    start: np.ndarray,
    regularization: float = 1e-9,
) -> np.ndarray:
    """
    Fit B-spline control points to a configuration-space path.

    A uniform cubic B-spline does not interpolate its control points: with
    control points p, the curve starts at (p_0 + 4 p_1 + p_2) / 6.  Using the
    path samples directly as control points therefore produces a guess whose
    first curve point is not `start`, i.e. one that violates the start equality
    constraint of the NLP.  Here the control points are instead obtained from a
    least-squares fit of the curve to the path, with curve(0) == start imposed
    as a hard equality constraint.

    Path samples are assumed to be uniformly spaced in the spline parameter.

    Parameters
    ----------
    path_points : np.ndarray
        Path samples of shape (num_path_samples, n_dof).
    num_control_points : int
        Number of control points of the resulting spline.
    start : np.ndarray
        Start configuration of shape (n_dof,), imposed exactly on curve(0).
    regularization : float, default=1e-9
        Tikhonov term keeping the normal equations non-singular.

    Raises
    ------
    ValueError
        If there are fewer path samples than control points, i.e. if the fit
        would be under-determined.

    Returns
    -------
    np.ndarray
        Control points of shape (num_control_points, n_dof).
    """
    path_points = np.asarray(path_points, dtype=float)
    start = np.asarray(start, dtype=float).reshape(1, -1)

    if path_points.ndim != 2:
        raise ValueError(
            f"path_points must have shape (num_path_samples, n_dof), got "
            f"{path_points.shape}."
        )

    if path_points.shape[0] < num_control_points:
        raise ValueError(
            f"{path_points.shape[0]} path samples cannot determine "
            f"{num_control_points} control points: the least-squares fit would "
            "be rank deficient and the result dominated by `regularization`. "
            "Interpolate the path to at least num_control_points samples first."
        )

    n_dof = path_points.shape[1]

    if start.shape[1] != n_dof:
        raise ValueError(
            f"start has {start.shape[1]} entries but the path has {n_dof} columns."
        )

    # Only the shape of the control points matters to build the basis.
    bspline = BSpline(np.zeros((num_control_points, n_dof)))

    basis = bspline._build_basis_matrix(
        bspline.sample_parameters(path_points.shape[0])
    )                                                    # (num_path_samples, N)
    start_basis = bspline.basis_matrix([0.0])            # (1, N)

    normal_matrix = basis.T @ basis + regularization * np.eye(num_control_points)

    # KKT system of  min ||basis @ P - path||^2  s.t.  start_basis @ P == start
    kkt_matrix = np.block(
        [
            [normal_matrix, start_basis.T],
            [start_basis, np.zeros((1, 1))],
        ]
    )
    kkt_rhs = np.vstack((basis.T @ path_points, start))

    solution = np.linalg.solve(kkt_matrix, kkt_rhs)

    return solution[:num_control_points, :]


class RRTStarInitializer(PathInitializer):
    """
    RRT* initializer in configuration space.

    The goal is defined in task space using:

        robot.f_task(q)

    Collision is checked using:

        robot.collision_points(q)

    This makes the initializer independent of whether the robot is a manipulator,
    drone, or future mobile robot.
    """

    def __init__(
        self,
        voxel_size: float = 0.10,
        goal_threshold: float = 0.01,
        range: float = 0.5,
        random_seed: Optional[int] = 42,
        check_interval: float = 0.2,
        max_time: float | None = None,
        unbounded_radius: float = _DEFAULT_UNBOUNDED_RADIUS,
        approximate_patience: int = _DEFAULT_APPROXIMATE_PATIENCE,
        improvement_tol: float = _DEFAULT_IMPROVEMENT_TOL,
    ):
        """
        Parameters
        ----------
        voxel_size:
            Resolution of the occupancy grid built from the obstacle means.
        goal_threshold:
            Task-space radius accepted as reaching the goal.  Keep it loose:
            the goal region is not sampleable (see TaskSpaceGoal), so RRT* has
            to hit it by chance.
        random_seed:
            Seed for OMPL's global RNG. None leaves it untouched.
        check_interval:
            Length of one planner.solve() slice, in seconds.
        max_time:
            Wall-clock budget for the search, in seconds.  None means
            `_DEFAULT_MAX_TIME`; it does NOT mean "unbounded".  When the budget
            expires the best approximate path found so far is returned and
            `InitializerResult.success` is False.
        unbounded_radius:
            Half-width of the box substituted for an infinite joint limit,
            centred on the start configuration.  Prefer declaring real limits
            on the robot over relying on this.
        approximate_patience:
            Give up after this many consecutive `check_interval` slices in
            which the approximate solution's distance to the goal did not
            improve.  This is what keeps a tight, unreachable goal_threshold
            from costing the whole `max_time` on every call.
        improvement_tol:
            Distance-to-goal improvement below which a slice counts as stalled.
        """
        self.voxel_size: float = voxel_size
        self.goal_threshold: float = goal_threshold
        self.check_interval: float = check_interval
        self.range: float = range
        self.max_time: float | None = max_time
        self.unbounded_radius: float = float(unbounded_radius)
        self.approximate_patience: int = int(approximate_patience)
        self.improvement_tol: float = float(improvement_tol)

        if random_seed is not None:
            ou.RNG.setSeed(random_seed)

    def initialize(
        self,
        robot,
        environment,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
    ) -> InitializerResult:
        t0 = perf_counter()

        self.robot = robot
        self.environment = environment
        self.n_dof = robot.n_dof
        self.joint_limits = self._joint_limits(robot, start)
        self.occupancy_map = self._build_occupancy_map(
            environment.obstacle_means_at(0),
            self.voxel_size,
        )

        space, space_info = self._setup_space()
        start_state = self._create_state(space, start)
        goal_region = TaskSpaceGoal(
            space_information=space_info,
            robot=robot,
            goal=np.asarray(goal, dtype=float),
            threshold=self.goal_threshold,
        )

        path_points, exact = self._solve_until_solution(
            space_info=space_info,
            start_state=start_state,
            goal_region=goal_region,
            num_control_points=num_control_points,
        )

        t1 = perf_counter()

        # `exact` is what `success` reports.  An approximate path -- the tree
        # node that happened to end up closest to the goal -- is still a much
        # better warm start than a constant guess, so it is used, but it must
        # never be reported as a solved planning problem.
        success = exact
        fitted = path_points is not None and path_points.shape[0] >= num_control_points

        if not fitted:
            # Constant guess: every control point equal, so curve(0) == start.
            control_points = np.tile(start, (num_control_points, 1))
        else:
            control_points = fit_control_points(
                path_points,
                num_control_points,
                start,
            )

        bspline = BSpline(control_points)
        trajectory = np.array(bspline.spline_eval(num_control_points))


        return InitializerResult(
            control_points=control_points,
            trajectory=trajectory,
            success=success,
            timings={"initializer": t1 - t0},
            metadata={
                "type": "rrtstar",
                "max_time": self.max_time,
                "check_interval": self.check_interval,
                "exact_solution": exact,
                "search_seconds": t1 - t0,
                "path_samples": 0 if path_points is None else int(path_points.shape[0]),
                "fitted_to_path": bool(fitted),
            },
        )
    
    def _solve_until_solution(
        self,
        space_info: ob.SpaceInformation,
        start_state: ob.State,
        goal_region: ob.GoalRegion,
        num_control_points: int,
    ) -> tuple[np.ndarray | None, bool]:
        """Run RRT* and return (path samples, exact).

        `exact` distinguishes a path that actually reaches the goal region from
        an approximate one.  The truthiness of OMPL's PlannerStatus does NOT:
        measured with ompl 1.7.0, an APPROXIMATE_SOLUTION gives
        `bool(status) == True`, so the previous `if solved: return path, True`
        reported the tree node that merely ended up closest to the goal as a
        solved problem.  `ProblemDefinition.hasExactSolution()` is the query
        that separates the two.

        The loop ends on the first of three conditions: an exact solution, an
        approximate solution that has stopped improving for
        `approximate_patience` slices, or the `max_time` budget.  The middle one
        is what keeps the honest reporting cheap; the last one is what makes
        termination unconditional, including when RRT* produces nothing at all
        (an invalid start state) -- the case where the old loop spun forever.
        """
        problem = ob.ProblemDefinition(space_info)
        problem.addStartState(start_state)
        problem.setGoal(goal_region)

        planner = og.RRTstar(space_info)
        planner.setRange(self.range)  
        planner.setProblemDefinition(problem)
        planner.setup()
        
        print("rage efective:", planner.getRange())

        budget = _DEFAULT_MAX_TIME if self.max_time is None else float(self.max_time)
        t0 = perf_counter()
        best_approximate = np.inf
        stalled_slices = 0

        while True:
            planner.solve(self.check_interval)

            exact = bool(problem.hasExactSolution())

            if not exact:
                if problem.hasSolution():
                    approximate = float(problem.getSolutionDifference())

                    if best_approximate - approximate > self.improvement_tol:
                        best_approximate = approximate
                        stalled_slices = 0
                    else:
                        stalled_slices += 1

                plateaued = stalled_slices >= self.approximate_patience
                expired = perf_counter() - t0 >= budget

                if not plateaued and not expired:
                    continue

            if not problem.hasSolution():
                return None, False

            path = problem.getSolutionPath()

            # PathGeometric.interpolate() only ever ADDS states, and does
            # nothing at all on a path of fewer than two states, so the caller
            # still has to check the returned count.
            path.interpolate(
                _PATH_SAMPLES_PER_CONTROL_POINT * num_control_points
            )
            return self._path_to_numpy(path), exact

    def _setup_space(self):
        space = ob.RealVectorStateSpace(self.n_dof)
        bounds = ob.RealVectorBounds(self.n_dof)

        for joint_idx, (lower, upper) in enumerate(self.joint_limits):
            bounds.setLow(joint_idx, float(lower))
            bounds.setHigh(joint_idx, float(upper))

        space.setBounds(bounds)

        space_info = ob.SpaceInformation(space)
        space_info.setStateValidityChecker(
            ob.StateValidityCheckerFn(self._is_state_valid)
        )
        space_info.setup()

        return space, space_info

    def _joint_limits(self, robot, start: np.ndarray) -> list[tuple[float, float]]:
        """
        Finite per-joint bounds for the OMPL state space.

        ob.RealVectorBounds cannot hold an infinity, so an unbounded joint has
        to be replaced by an artificial box.  It is centred on the start
        configuration and only `unbounded_radius` wide, NOT the +-1e6 this used
        to substitute: RRT* samples its state space uniformly, so a 2e6-wide
        box makes every sample land astronomically far from the scene and the
        tree never approaches the goal.  Substituting a bound is still a guess,
        so it warns -- declare real limits on the robot instead.
        """
        start = np.asarray(start, dtype=float).reshape(-1)

        if start.shape[0] != robot.n_dof:
            raise ValueError(
                f"start has {start.shape[0]} entries for n_dof={robot.n_dof}."
            )

        limits = robot.joint_limits()

        if limits is None:
            limits = [(-np.inf, np.inf) for _ in range(robot.n_dof)]

        limits = list(limits)

        if len(limits) != robot.n_dof:
            raise ValueError(
                f"joint_limits must contain {robot.n_dof} entries, "
                f"got {len(limits)}."
            )

        clean_limits = []
        substituted = []

        for joint_idx, (lower, upper) in enumerate(limits):
            lower = float(lower)
            upper = float(upper)
            was_infinite = False

            if np.isneginf(lower):
                lower = start[joint_idx] - self.unbounded_radius
                was_infinite = True

            if np.isposinf(upper):
                upper = start[joint_idx] + self.unbounded_radius
                was_infinite = True

            if was_infinite:
                substituted.append(joint_idx)

            clean_limits.append((lower, upper))

        if substituted:
            warnings.warn(
                f"{type(robot).__name__} declares no finite limits for joints "
                f"{substituted}; RRT* will sample "
                f"start +- {self.unbounded_radius} on them. Pass explicit "
                "limits (e.g. xyz_limits / xy_limits) covering the scene, "
                "otherwise the initial guess will be poor.",
                RuntimeWarning,
                stacklevel=2,
            )

        return clean_limits

    def _create_state(self, space, q: np.ndarray):
        state = ob.State(space)

        for joint_idx in range(self.n_dof):
            state[joint_idx] = float(q[joint_idx])

        return state

    def _is_state_valid(self, state) -> bool:
        q = np.array(
            [state[joint_idx] for joint_idx in range(self.n_dof)],
            dtype=float,
        )

        if not self._within_joint_limits(q):
            return False

        if self._robot_in_collision(q):
            return False

        return True

    def _within_joint_limits(self, q: np.ndarray) -> bool:
        for joint_idx, (lower, upper) in enumerate(self.joint_limits):
            if q[joint_idx] < lower or q[joint_idx] > upper:
                return False

        return True

    def _robot_in_collision(self, q: np.ndarray) -> bool:
        points = self.robot.collision_points(q)

        if hasattr(points, "full"):
            points = points.full()

        points = np.asarray(points, dtype=float).reshape(-1, 3)

        included = self.occupancy_map.check_if_included(
            o3d.utility.Vector3dVector(points)
        )

        return bool(np.any(np.asarray(included, dtype=bool)))

    def _build_occupancy_map(self, obstacle_means, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obstacle_means)

        return o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd,
            voxel_size=voxel_size,
        )

    def _path_to_numpy(self, path: og.PathGeometric) -> np.ndarray:
        states = []

        for path_idx in range(path.getStateCount()):
            state = path.getState(path_idx)

            q = np.array(
                [float(state[joint_idx]) for joint_idx in range(self.n_dof)],
                dtype=float,
            )

            states.append(q)

        return np.asarray(states, dtype=float)

class TaskSpaceGoal(ob.GoalRegion):
    """
    OMPL goal region defined through robot.f_task(q).

    This is an ob.GoalRegion and NOT an ob.GoalSampleableRegion: sampling a
    configuration that puts f_task(q) on the goal would need inverse
    kinematics.  RRT* therefore cannot goal-bias towards it and only satisfies
    it by landing inside `threshold` by chance, which is why the search has to
    be time-bounded and why an approximate result is the normal outcome for a
    tight threshold.
    """

    def __init__(
        self,
        space_information,
        robot,
        goal: np.ndarray,
        threshold: float = 0.01,
    ):
        super().__init__(space_information)

        self.robot = robot
        self.goal = np.asarray(goal, dtype=float).reshape(3)

        self.setThreshold(float(threshold))

    def distanceGoal(self, state) -> float:
        q = np.array(
            [state[i] for i in range(self.robot.n_dof)],
            dtype=float,
        )

        point = self.robot.f_task(q)

        if hasattr(point, "full"):
            point = point.full()

        point = np.asarray(point, dtype=float).reshape(3)

        return float(np.linalg.norm(point - self.goal))