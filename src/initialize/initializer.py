from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class InitializerResult:
    """
    Standard output for all initializers.
    """

    control_points: np.ndarray
    trajectory: np.ndarray | None = None
    success: bool = True
    timings: dict[str, float] | None = None
    metadata: dict | None = None

class PathInitializer(ABC):
    """
    Common interface for trajectory initializers.
    """

    @abstractmethod
    def initialize(
        self,
        robot,
        environment,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
    ) -> InitializerResult:
        raise NotImplementedError