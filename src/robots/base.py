from abc import ABC, abstractmethod
from typing import Any

class BaseRobot(ABC):
    """
    Minimal common interface for all robot models.
    """

    # Set to True by robots whose collision geometry depends on the second
    # derivative of q (e.g. an underactuated drone whose roll/pitch follow
    # commanded acceleration). Callers that have an analytic or
    # finite-difference qdd estimate available check this flag to decide
    # whether it's worth computing and passing one; robots that ignore qdd
    # don't need it.
    requires_acceleration: bool = False

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of optimization variables."""
        raise NotImplementedError

    @abstractmethod
    def f_task(self, q: Any) -> Any:
        """
        Map configuration q to the task-space point used by the goal cost.
        """
        raise NotImplementedError

    @abstractmethod
    def collision_points(self, q: Any, qdd: Any = None) -> Any:
        """
        Return collision evaluation points with shape (n_points, 3).

        qdd, the acceleration (d^2q/dt^2) at this configuration, is optional
        and only meaningful to robots with requires_acceleration = True.
        """
        raise NotImplementedError


    @abstractmethod
    def forward_kinematics(self, q: Any, qdd: Any = None) -> Any:
        """
        Return forward kinematics of the robot.
        """
        raise NotImplementedError
    
    @abstractmethod
    def collision_covariances(self): 
        raise NotImplementedError

    @abstractmethod
    def collision_covariances_online(self, q: Any, qdd: Any = None) -> Any:
        raise NotImplementedError

    def joint_limits(self) -> list[tuple[float, float]] | None:
        """
        Optional joint limits.

        Robots without limits can return None.
        """
        return None
    
    def validate_q(self, q: Any) -> None:
        if q.shape[0] != self.n_dof:
            raise ValueError(
                f"Expected q of length {self.n_dof}, got {q.shape[0]}"
            )
    