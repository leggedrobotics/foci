import numpy as np

from abc import ABC, abstractmethod


class BaseObstacle(ABC):
    """
    Minimal common interface for obstacle models.
    """

    @abstractmethod
    def obstacle_mean(self, k: int) -> np.ndarray:
        """Return the obstacle mean at sample index k. Shape: (3,)."""
        raise NotImplementedError

    @abstractmethod
    def obstacle_covariance(self, k: int) -> np.ndarray:
        """Return the obstacle covariance at sample index k. Shape: (3, 3)."""
        raise NotImplementedError


class StaticObstacle(BaseObstacle):
    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        self._mean = np.asarray(mean, dtype=float)
        self._covariance = np.asarray(covariance, dtype=float)

        if self._mean.shape != (3,):
            raise ValueError("mean must have shape (3,).")
        if self._covariance.shape != (3, 3):
            raise ValueError("covariance must have shape (3, 3).")

    def obstacle_mean(self, k: int) -> np.ndarray:
        return self._mean

    def obstacle_covariance(self, k: int) -> np.ndarray:
        return self._covariance


class MobileObstacle(BaseObstacle):
    """
    Obstacle whose position shifts by a pre-computed trajectory offset.

    trajectory: shape (num_samples, 3)
        trajectory[k] is the position offset at sample index k.
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        trajectory: np.ndarray,
    ):
        self._mean = np.asarray(mean, dtype=float)
        self._covariance = np.asarray(covariance, dtype=float)
        self.trajectory = np.asarray(trajectory, dtype=float)

        if self._mean.shape != (3,):
            raise ValueError("mean must have shape (3,).")
        if self._covariance.shape != (3, 3):
            raise ValueError("covariance must have shape (3, 3).")
        if self.trajectory.ndim != 2 or self.trajectory.shape[1] != 3:
            raise ValueError("trajectory must have shape (num_samples, 3).")

    def obstacle_mean(self, k: int) -> np.ndarray:
        return self._mean + self.trajectory[k]

    def obstacle_covariance(self, k: int) -> np.ndarray:
        return self._covariance