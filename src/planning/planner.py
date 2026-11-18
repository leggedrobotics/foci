"""Static trajectory optimisation planner.

Uses a pre-built environment whose obstacle statistics do not change between
planning calls.
"""

from __future__ import annotations

from src.optimization.problem import build_problem
from src.planning.joints import JointGroups

from src.planning.base import BasePlanner


class Planner(BasePlanner):
    """Trajectory optimisation planner for static environments.

    Obstacle means and covariances are fixed at construction time and remain
    constant across all calls to plan().

    See BasePlanner for parameter documentation.
    """

    # Safe here: unlike OnlinePlanner, obstacle means/covariances are fixed
    # at construction and never baked per-call into the symbolic graph, so
    # the built NLP and IPOPT solver can be reused across plan() calls --
    # see BasePlanner._cache_built_problem.
    _cache_built_problem = True

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