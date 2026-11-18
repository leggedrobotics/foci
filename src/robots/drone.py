from dataclasses import dataclass, field
from typing import Any

import casadi as cas
import numpy as np

from .base import BaseRobot


def _quat_mul(q1: Any, q2: Any) -> Any:
    """Hamilton product q1 * q2, both [w, x, y, z] (scalar-first)."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return cas.vertcat(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_to_rot(q: Any) -> Any:
    """Rotation matrix from unit quaternion q = [w, x, y, z]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return cas.vertcat(
        cas.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)),
        cas.horzcat(2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        cas.horzcat(2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)),
    )


def _rotation_matrix_from_acceleration(a: Any, yaw: Any, g: float = 9.81) -> Any:
    """
    Rotation matrix R_b^w of a standard (underactuated) multirotor from its
    commanded horizontal acceleration and yaw.

    a:   3-vector (ax, ay, az), the body's commanded world-frame acceleration.
    yaw: yaw angle [rad].
    g:   gravity magnitude [m/s^2].

    The airframe tilts so that its thrust axis points along
    xi = a + [0, 0, g] (the only direction a single rotor-thrust axis can
    produce that acceleration), then spins about that axis by yaw. a = 0
    (level flight) makes xi = [0, 0, g], the tilt quaternion collapses to the
    identity, and this reduces exactly to a yaw-only rotation.
    """
    xi = cas.vertcat(a[0], a[1], a[2] + g)
    xin = xi / cas.norm_2(xi)

    # Hopf-fibration tilt quaternion (scalar-first); 1/sqrt(2(1+xin_z)) makes
    # it unit-norm analytically, no extra normalization needed. Guarded
    # against xin_z -> -1 (free-fall straight down), where the tilt is
    # undefined.
    wx = cas.fmax(1.0 + xin[2], 1e-6)
    q_xi = cas.vertcat(wx, -xin[1], xin[0], 0.0) / cas.sqrt(2.0 * wx)

    q_psi = cas.vertcat(cas.cos(yaw / 2), 0.0, 0.0, cas.sin(yaw / 2))

    q_bw = _quat_mul(q_xi, q_psi)
    return _quat_to_rot(q_bw)


def finite_difference_acceleration(trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Central-difference acceleration estimate for a finished (q over time)
    trajectory array, shape (num_samples, n_dof) in, same shape out.

    Endpoints reuse their nearest interior estimate (one-sided differences
    would otherwise need a 4th point). This is a best-effort estimate for
    evaluating an already-produced trajectory (visualization, metrics) -- not
    for differentiating through an optimizer, where it would need to account
    for every sample's coupling to its neighbours.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    num_samples = trajectory.shape[0]

    if num_samples < 3:
        return np.zeros_like(trajectory)

    accel = np.zeros_like(trajectory)
    accel[1:-1] = (trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]) / dt**2
    accel[0] = accel[1]
    accel[-1] = accel[-2]
    return accel


@dataclass(frozen=True)
class DroneGaussian:
    """
    Gaussian collision point attached to the drone body.

    Parameters
    ----------
    offset:
        Local body-frame offset from the drone center.

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


class DroneRobot(BaseRobot):
    """
    Omnidirectional drone model with state:

        q = [x, y, z, yaw]

    The task point is the drone center [x, y, z].

    Collision is represented by body-frame Gaussian points. By default:
    - center
    - left arm endpoint
    - right arm endpoint

    Fully-actuated / omnidirectional: xyz translation and yaw are
    independent actuator channels (e.g. a tilt-rotor hex/octocopter), so the
    body never needs to tilt to accelerate -- collision geometry only ever
    rotates about z, with the world-frame qdd ignored. See
    StandardDroneRobot for an underactuated multirotor, whose roll/pitch
    follow its commanded acceleration instead.
    """

    def __init__(
        self,
        urdf_path: str,
        arm_length: float = 0.15,
        gaussian_specs: list[DroneGaussian | dict | tuple] | None = None,
        xyz_limits: list[tuple[float, float]] =  [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)],
        yaw_limits: tuple[float, float] = (-np.pi, np.pi),
    ):
        if arm_length <= 0:
            raise ValueError("arm_length must be positive.")

        self._urdf_path = urdf_path
        self.arm_length = float(arm_length)
        self.gaussian_specs = self._parse_gaussian_specs(gaussian_specs)

        self.xyz_limits = xyz_limits
        self.yaw_limits = yaw_limits

    @property
    def n_dof(self) -> int:
        return 4
    
    @property
    def urdf_path(self) -> str:
        return self._urdf_path


    def f_task(self, q: Any) -> Any:
        """
        Drone task point: center position [x, y, z].
        """
        return cas.vertcat(q[0], q[1], q[2])

    def forward_kinematics(self, q: Any, qdd: Any = None) -> Any:
        """
        For the drone, FK is simply the collision points.
        """
        return self.collision_points(q, qdd)

    def _body_rotation(self, q: Any, qdd: Any = None) -> Any:
        """
        World-frame rotation matrix R_b^w of the body, used for both
        collision_points and collision_covariances_online so the two always
        agree. qdd is ignored: an omnidirectional drone's body never tilts,
        it only ever spins about z.
        """
        yaw = q[3]
        c, s = cas.cos(yaw), cas.sin(yaw)
        return cas.vertcat(
            cas.horzcat(c, -s, 0),
            cas.horzcat(s, c, 0),
            cas.horzcat(0, 0, 1),
        )

    def collision_points(self, q: Any, qdd: Any = None) -> Any:
        """
        Return world-frame Gaussian centers.

        qdd is ignored: an omnidirectional drone's body never tilts.

        Shape:
            (n_gaussians, 3)
        """
        center = cas.vertcat(q[0], q[1], q[2])
        R = self._body_rotation(q, qdd)

        points = []

        for spec in self.gaussian_specs:
            world_offset = R @ cas.DM(spec.offset)
            point = center + world_offset
            points.append(point.T)

        return cas.vertcat(*points)

    def collision_covariances(self) -> np.ndarray:
        """
        Return one covariance matrix per drone Gaussian.

        Shape:
            (n_gaussians, 3, 3)
        """
        return np.stack(
            [spec.covariance for spec in self.gaussian_specs],
            axis=0,
        )

    def collision_covariances_online(self, q: Any, qdd: Any = None) -> Any:
        """qdd is ignored: an omnidirectional drone's body never tilts."""

        R = self._body_rotation(q, qdd)
        covs = []

        for spec in self.gaussian_specs: 
            body_cov = cas.DM(spec.covariance)
            world_cov = R @ body_cov @ R.T

            covs.append(world_cov)

        return cas.vertcat(*covs)


    def joint_limits(self) -> list[tuple[float, float]]:
        return [
            *self.xyz_limits,
            self.yaw_limits,
        ]

    def _parse_gaussian_specs(
        self,
        gaussian_specs: list[DroneGaussian | dict | tuple] | None,
    ) -> list[DroneGaussian]:
        """
        Normalize user-defined drone Gaussian specs.
        """
        if gaussian_specs is None:
            return [
                DroneGaussian(
                    offset=np.array([0.0, 0.0, 0.0]),
                    covariance=np.eye(3) * 0.1**2,
                    name="center",
                ),
            ]

        parsed = []

        for idx, spec in enumerate(gaussian_specs):
            if isinstance(spec, DroneGaussian):
                gaussian = spec

            elif isinstance(spec, dict):
                gaussian = DroneGaussian(
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
                        "Drone Gaussian tuple must be "
                        "(offset,) or (offset, covariance)."
                    )

                gaussian = DroneGaussian(
                    offset=np.asarray(offset, dtype=float),
                    covariance=np.asarray(covariance, dtype=float),
                    name=f"gaussian_{idx}",
                )

            parsed.append(gaussian)

        return parsed


class StandardDroneRobot(DroneRobot):
    """
    Standard (underactuated) multirotor with the same state as DroneRobot:

        q = [x, y, z, yaw]

    Unlike DroneRobot, horizontal translation here is only achievable by
    tilting the airframe: a single thrust axis can't produce an arbitrary
    xyz acceleration on its own, so roll/pitch aren't free -- they're
    whatever the commanded acceleration (plus gravity) and yaw require. See
    _rotation_matrix_from_acceleration.

    collision_points/collision_covariances_online therefore take an optional
    qdd (the acceleration at this q): when it's available (e.g. the FOCI
    NLP's own spline second derivative, or a finite-difference estimate from
    a finished trajectory) the body tilts accordingly; when it's None (e.g.
    the RRT* initializer, which only ever sees one configuration at a time
    with no derivative context) this falls back to level flight -- the same
    yaw-only rotation as DroneRobot.
    """

    requires_acceleration = True

    def __init__(self, *args, g: float = 9.81, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = float(g)

    def _body_rotation(self, q: Any, qdd: Any = None) -> Any:
        """
        World-frame rotation matrix, tilted by the acceleration implied by
        qdd (its first 3 components; yaw'' plays no role here). qdd = None
        falls back to a = 0 (level flight), identical to DroneRobot's
        yaw-only rotation -- see _rotation_matrix_from_acceleration.
        """
        yaw = q[3]
        a = qdd[0:3] if qdd is not None else cas.DM.zeros(3)
        return _rotation_matrix_from_acceleration(a, yaw, self.g)