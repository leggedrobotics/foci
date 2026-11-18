from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlanningResult:
    """
    Standard output container for all planners.

    This class is shared across:
    - FOCI
    - CHOMP
    - STOMP
    - future planners

    Attributes
    ----------
    trajectory :
        Dense optimized trajectory of shape (N, n_dof).

    control_points :
        Optimized spline control points.

    initial_trajectory :
        Optional initializer trajectory.

    success :
        Whether the optimization succeeded.

    timings :
        Dictionary with execution times.

    solver_stats :
        Raw solver statistics.

    metadata :
        Optional planner-specific information.
    """

    trajectory: np.ndarray

    control_points: np.ndarray | None = None

    initial_trajectory: np.ndarray | None = None

    success: bool = True

    timings: dict[str, float] = field(default_factory=dict)

    solver_stats: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        """
        Return total planning time if available.
        """
        return self.timings.get("total")

    @property
    def n_samples(self) -> int:
        return self.trajectory.shape[0]

    @property
    def n_dof(self) -> int:
        return self.trajectory.shape[1]

    def final_configuration(self) -> np.ndarray:
        return self.trajectory[-1]

    def initial_configuration(self) -> np.ndarray:
        return self.trajectory[0]