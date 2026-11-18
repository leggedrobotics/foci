from __future__ import annotations

import numpy as np

from src.initialize.initializer import InitializerResult, PathInitializer

class UniformControlPointsInitializer(PathInitializer):
    """
    Simple fallback initializer using linear interpolation in configuration space.
    """

    def initialize(
        self,
        robot,
        environment,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
    ) -> InitializerResult:
        

        start = np.asarray(start, dtype=float)
        control_points = np.tile(start, (num_control_points, 1))

        return InitializerResult(
            control_points=control_points,
            trajectory=control_points,
            success=True,
            timings={"initializer": 0.0},
            metadata={"type": "straight_line_constant"},
        )