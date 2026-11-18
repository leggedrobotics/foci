from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.environment.obstacle import BaseObstacle



class BaseGaussianEnvironment(ABC):
    """Abstract Gaussian obstacle environment.

    An environment provides obstacle means and covariances, optionally
    varying over discrete time samples k = 0, 1, …, K.

    Subclasses must implement
        obstacle_means_at(k)       -> (n_obstacles, 3)
        obstacle_covariances_at(k) -> (n_obstacles, 3, 3)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_obstacles(self) -> int:
        """Number of obstacles in the environment."""

    @abstractmethod
    def obstacle_means_at(self, k: int) -> np.ndarray:
        """Return obstacle means at sample k.

        Parameters
        ----------
        k:
            Discrete time index.

        Returns
        -------
        np.ndarray, shape (n_obstacles, 3)
        """

    @abstractmethod
    def obstacle_covariances_at(self, k: int) -> np.ndarray:
        """Return obstacle covariances at sample k.

        Parameters
        ----------
        k:
            Discrete time index.

        Returns
        -------
        np.ndarray, shape (n_obstacles, 3, 3)
        """

    # ------------------------------------------------------------------
    # Shared derived operations
    # ------------------------------------------------------------------

    def convolved_covariances_at(
        self,
        k: int,
        robot_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return determinants and inverses of (obstacle + robot) covariances.

        Parameters
        ----------
        k:
            Discrete time index.
        robot_covariance:
            Shape (3, 3). Added to every obstacle covariance before inversion.

        Returns
        -------
        determinants : np.ndarray, shape (n_obstacles,)
        inverses     : np.ndarray, shape (n_obstacles, 3, 3)
        """
        robot_covariance = np.asarray(robot_covariance, dtype=float)
        if robot_covariance.shape != (3, 3):
            raise ValueError("robot_covariance must have shape (3, 3).")

        covariances = self.obstacle_covariances_at(k) + robot_covariance
        return np.linalg.det(covariances), np.linalg.inv(covariances)

    def build_per_sample_means(self, num_samples: int) -> np.ndarray:
        """Collect obstacle means for all samples into a single array.

        Returns
        -------
        np.ndarray, shape (num_samples, n_obstacles * 3)
            Row k contains the flattened means of all obstacles at sample k:
            [mean0_x, mean0_y, mean0_z, mean1_x, …]
        """
        return np.stack(
            [self.obstacle_means_at(k).reshape(-1) for k in range(num_samples)],
            axis=0,
        )

    def build_per_sample_covariances(self, num_samples: int) -> np.ndarray:
        """Collect obstacle covariances for all samples into a single array.

        Returns
        -------
        np.ndarray, shape (num_samples, n_obstacles * 9)
            Row k contains the flattened covariances of all obstacles at sample k:
            [cov0_00, cov0_01, …, cov0_22, cov1_00, …]
        """
        return np.stack(
            [self.obstacle_covariances_at(k).reshape(-1) for k in range(num_samples)],
            axis=0,
        )

@dataclass
class GaussianEnvironment(BaseGaussianEnvironment):
    """Gaussian obstacle environment with static (time-invariant) obstacles.
 
    Parameters
    ----------
    obstacle_means:
        Shape (n_obstacles, 3).
    obstacle_covariances:
        Shape (n_obstacles, 3, 3).
    """
 
    obstacle_means: np.ndarray
    obstacle_covariances: np.ndarray
 
    def __post_init__(self) -> None:
        self.obstacle_means = np.asarray(self.obstacle_means, dtype=float)
        self.obstacle_covariances = np.asarray(self.obstacle_covariances, dtype=float)
 
        if self.obstacle_means.ndim != 2 or self.obstacle_means.shape[1] != 3:
            raise ValueError("obstacle_means must have shape (n_obstacles, 3).")
 
        if (
            self.obstacle_covariances.ndim != 3
            or self.obstacle_covariances.shape[1:] != (3, 3)
        ):
            raise ValueError(
                "obstacle_covariances must have shape (n_obstacles, 3, 3)."
            )
 
        if self.obstacle_covariances.shape[0] != self.obstacle_means.shape[0]:
            raise ValueError(
                "obstacle_means and obstacle_covariances must contain "
                "the same number of obstacles."
            )
 
    # ------------------------------------------------------------------
    # BaseGaussianEnvironment interface
    # ------------------------------------------------------------------
 
    @property
    def n_obstacles(self) -> int:
        return self.obstacle_means.shape[0]
 
    def obstacle_means_at(self, k: int) -> np.ndarray:  # noqa: ARG002
        """Return obstacle means (identical for every k).
 
        Returns
        -------
        np.ndarray, shape (n_obstacles, 3)
        """
        return self.obstacle_means
 
    def obstacle_covariances_at(self, k: int) -> np.ndarray:  # noqa: ARG002
        """Return obstacle covariances (identical for every k).
 
        Returns
        -------
        np.ndarray, shape (n_obstacles, 3, 3)
        """
        return self.obstacle_covariances



@dataclass
class GaussianEnvironmentOnline(BaseGaussianEnvironment):
    """Gaussian obstacle environment supporting static and mobile obstacles.

    Parameters
    ----------
    obstacles:
        Each element implements the BaseObstacle protocol:
            obstacle_mean(k)       -> (3,)
            obstacle_covariance(k) -> (3, 3)
    """

    obstacles: list[BaseObstacle]

    # ------------------------------------------------------------------
    # BaseGaussianEnvironment interface
    # ------------------------------------------------------------------

    @property
    def n_obstacles(self) -> int:
        return len(self.obstacles)

    def obstacle_means_at(self, k: int) -> np.ndarray:
        """Return obstacle means at sample k.

        Returns
        -------
        np.ndarray, shape (n_obstacles, 3)
        """
        return np.stack([o.obstacle_mean(k) for o in self.obstacles], axis=0)

    def obstacle_covariances_at(self, k: int) -> np.ndarray:
        """Return obstacle covariances at sample k.

        Returns
        -------
        np.ndarray, shape (n_obstacles, 3, 3)
        """
        return np.stack([o.obstacle_covariance(k) for o in self.obstacles], axis=0)