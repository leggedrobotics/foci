from dataclasses import dataclass, field
from typing import Any

import casadi as cas
import numpy as np

from .base import BaseRobot


@dataclass(frozen=True)
class MobileGaussian:
    """
    Gaussian collision point attached to the mobile robot body.

    Parameters
    ----------
    offset:
        Local body-frame offset from the robot center.

    covariance:
        Gaussian covariance matrix.

    name:
        Optional human-readable name.
    """

    offset: np.ndarray
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.1**2
    )
    name: str | None = None

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset, dtype=float).reshape(3)
        covariance = np.asarray(self.covariance, dtype=float)

        if covariance.shape != (3, 3):
            raise ValueError(
                f"Drone Gaussian covariance must have shape (3, 3), "
                f"got {covariance.shape}."
            )

        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "covariance", covariance)


class MobileRobot(BaseRobot):
    """
    Drone model with state:

        q = [x, y, yaw]

    The task point is the mobile center [x, y, z].
    """

    def __init__(
        self,
        urdf_path: str,
        body_center_height: float = 1.0,
        gaussian_specs: list[MobileGaussian | dict | tuple] | None = None,
        xy_limits: list[tuple[float, float]] =  [(-np.inf, np.inf), (-np.inf, np.inf)],
        yaw_limits: tuple[float, float] = (-np.pi, np.pi),
    ):
        if body_center_height <= 0:
            raise ValueError("body_center_height must be positive.")

        self._urdf_path = urdf_path
        self.body_center_height = float(body_center_height)
        self.gaussian_specs = self._parse_gaussian_specs(gaussian_specs)

        self.xy_limits = xy_limits
        self.yaw_limits = yaw_limits

    @property
    def n_dof(self) -> int:
        return 3
    
    @property
    def urdf_path(self) -> str:
        return self._urdf_path


    def f_task(self, q: Any) -> Any:
        """
        Mobile task point: center position [x, y, z].
        """
        return cas.vertcat(q[0], q[1], self.body_center_height)

    def forward_kinematics(self, q: Any, qdd: Any = None) -> Any:
        """
        For the mobile robot, FK is simply the collision points.
        """
        return self.collision_points(q)

    def collision_points(self, q: Any, qdd: Any = None) -> Any:
        """
        Return world-frame Gaussian centers. qdd is ignored (requires_acceleration = False).

        Shape:
            (n_gaussians, 3)
        """
        x, y, yaw = q[0], q[1], q[2]

        c = cas.cos(yaw)
        s = cas.sin(yaw)

        center = cas.vertcat(x, y, self.body_center_height)

        points = []

        for spec in self.gaussian_specs:
            ox, oy, oz = spec.offset

            # Rotation around z-axis.
            world_offset = cas.vertcat(
                c * ox - s * oy,
                s * ox + c * oy,
                oz,
            )

            point = center + world_offset
            points.append(point.T)

        return cas.vertcat(*points)

    def collision_covariances(self) -> np.ndarray:
        """
        Return one covariance matrix per mobile Gaussian.

        Shape:
            (n_gaussians, 3, 3)
        """
        return np.stack(
            [spec.covariance for spec in self.gaussian_specs],
            axis=0,
        )

    def collision_covariances_online(self, q, qdd=None):
            yaw = q[2]
            
            c, s = cas.cos(yaw), cas.sin(yaw)
            R = cas.vertcat(
                cas.horzcat(c, -s, 0), 
                cas.horzcat(s, c, 0), 
                cas.horzcat(0, 0, 1), 
            )
    
            covs = []
    
            for spec in self.gaussian_specs: 
                body_cov = cas.DM(spec.covariance)
                world_cov = R @ body_cov @ R.T
    
                covs.append(world_cov)
    
            return cas.vertcat(*covs)

    def joint_limits(self) -> list[tuple[float, float]]:
        return [
            *self.xy_limits,
            self.yaw_limits,
        ]

    def _parse_gaussian_specs(
            self,
            gaussian_specs: list[MobileGaussian | dict | tuple] | None,
        ) -> list[MobileGaussian]:
            """
            Normalize user-defined mobile Gaussian specs.
            """
            if gaussian_specs is None:
                return [
                    MobileGaussian(
                        offset=np.array([0.0, 0.0, 0.0]),
                        covariance=np.eye(3) * 0.1**2,
                        name="center",
                    ),
                ]
    
            parsed = []
    
            for idx, spec in enumerate(gaussian_specs):
                if isinstance(spec, MobileGaussian):
                    gaussian = spec
    
                elif isinstance(spec, dict):
                    gaussian = MobileGaussian(
                        offset=np.asarray(spec["offset"], dtype=float),
                        covariance=np.asarray(
                            spec.get("covariance", np.eye(3) * 0.1**2),
                            dtype=float,
                        ),
                        name=spec.get("name", f"gaussian_{idx}"),
                    )
    
                else:
                    if len(spec) == 1:
                        offset = spec[0]
                        covariance = np.eye(3) * 0.1**2
                    elif len(spec) == 2:
                        offset, covariance = spec
                    else:
                        raise ValueError(
                            "Mobile Gaussian tuple must be "
                            "(offset,) or (offset, covariance)."
                        )
    
                    gaussian = MobileGaussian(
                        offset=np.asarray(offset, dtype=float),
                        covariance=np.asarray(covariance, dtype=float),
                        name=f"gaussian_{idx}",
                    )
    
                parsed.append(gaussian)
    
            return parsed