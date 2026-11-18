"""Online trajectory optimisation planner.

Uses an environment that may query obstacle statistics per time step,
suitable for dynamic or time-varying obstacle scenes.
"""

from __future__ import annotations

from src.optimization.problem_online import build_problem

from src.planning.base import BasePlanner


class OnlinePlanner(BasePlanner):
    """Trajectory optimisation planner for online (dynamic) environments.

    Obstacle means and covariances are re-queried at each sample index k
    during problem construction, allowing mobile obstacles to be accounted for.

    See BasePlanner for parameter documentation.
    """

    def _build_problem(self) -> tuple:
        return build_problem(
            robot=self.robot,
            environment=self.environment,
            num_control_points=self.num_control_points,
            num_samples=self.num_samples,
            joint_groups=self.joint_groups,
            weights=self.weights,
            vmax=self.vmax,
        )