"""
Correctness tests for StandardDroneRobot's acceleration-coupled tilt.

Two independent things are checked:

1. The rotation itself (collision_points / collision_covariances_online):
   physical properties of the tilt (thrust-axis alignment, reduction to a
   pure yaw rotation at zero acceleration, spectrum preservation) are
   checked against a geometric definition written independently in this
   file -- not against drone.py's own formula, which would just be a
   tautology.

2. That the tilt actually reaches the planner: the online NLP's obstacle
   cost (src/optimization/problem_online.py) is evaluated directly (no
   solve needed) on a hand-built trajectory with a known lateral
   acceleration, for an obstacle placed so that only the TILTED Gaussian
   footprint reaches it. A tilt-blind model (DroneRobot) must see ~zero
   cost there; StandardDroneRobot, with the same geometry, must see a
   clearly nonzero one.
"""

import numpy as np
import casadi as cas
import pytest

from src.robots.drone import DroneRobot, StandardDroneRobot, DroneGaussian
from src.environment.environment import GaussianEnvironmentOnline
from src.environment.obstacle import StaticObstacle
from src.optimization.utils import build_collision_point_map
from src.optimization.problem_online import _build_collision_cov_map, _obstacle_cost


G = 9.81


def _minimal_rotation_z_to(target_unit: np.ndarray) -> np.ndarray:
    """Rodrigues' formula for the shortest rotation taking [0,0,1] to target_unit."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, target_unit)
    s = np.linalg.norm(v)
    c = float(np.dot(z, target_unit))

    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _reference_rotation(a: np.ndarray, yaw: float, g: float = G) -> np.ndarray:
    """
    Independent (Rodrigues axis-angle, not quaternion) construction of the
    same physical rotation: spin the heading by yaw about world z, then
    apply the minimal rotation that takes the body z-axis to the thrust
    direction xi = a + [0, 0, g] to that spun heading as well.

    This is geometrically the same rotation drone.py's Hopf-fibration
    quaternion computes (verified numerically against it while writing this
    test), but derived a completely different way, so it is a real check
    rather than restating the same formula.
    """
    xi = np.asarray(a, dtype=float) + np.array([0.0, 0.0, g])
    z_axis = xi / np.linalg.norm(xi)

    tilt = _minimal_rotation_z_to(z_axis)
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x_axis = tilt @ heading

    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)  # columns = body axes in world frame


def _drone_fn(robot):
    """CasADi function (q, qdd) -> (points, covariances) for a drone robot."""
    q = cas.MX.sym("q", 4)
    qdd = cas.MX.sym("qdd", 4)
    points = robot.collision_points(q, qdd)
    covs = robot.collision_covariances_online(q, qdd)
    return cas.Function("drone_geom", [q, qdd], [points, covs])


# ---------------------------------------------------------------------------
# 1. Rotation correctness
# ---------------------------------------------------------------------------


def test_zero_acceleration_matches_yaw_only_omni_robot():
    """qdd = 0 must reduce StandardDroneRobot exactly to DroneRobot's yaw-only rotation."""
    specs = [
        DroneGaussian(np.array([0.3, 0.0, 0.0]), np.diag([0.2, 0.05, 0.05])),
        DroneGaussian(np.array([-0.3, 0.0, 0.0]), np.diag([0.2, 0.05, 0.05])),
    ]
    omni = DroneRobot(urdf_path=None, gaussian_specs=specs)
    standard = StandardDroneRobot(urdf_path=None, gaussian_specs=specs)

    q = np.array([1.0, -2.0, 0.5, 0.7])
    qdd_zero = np.zeros(4)

    pts_omni = np.asarray(omni.collision_points(q), dtype=float)
    pts_std_zero = np.asarray(standard.collision_points(q, qdd_zero), dtype=float)
    pts_std_none = np.asarray(standard.collision_points(q, None), dtype=float)

    assert np.allclose(pts_omni, pts_std_zero, atol=1e-10)
    assert np.allclose(pts_omni, pts_std_none, atol=1e-10)

    covs_omni = np.asarray(omni.collision_covariances_online(q), dtype=float)
    covs_std_zero = np.asarray(standard.collision_covariances_online(q, qdd_zero), dtype=float)

    assert np.allclose(covs_omni, covs_std_zero, atol=1e-10)


@pytest.mark.parametrize(
    "a, yaw",
    [
        (np.array([0.0, 4.0, 0.0]), 0.0),
        (np.array([3.0, 0.0, 0.0]), 0.0),
        (np.array([2.0, -3.0, 1.0]), 0.4),
        (np.array([-5.0, 1.5, 0.0]), -1.1),
    ],
)
def test_rotation_matches_independent_geometric_definition(a, yaw):
    """
    collision_covariances_online's rotation, read back out of a known
    diagonal covariance, must match an independently-derived rotation
    matrix (thrust-axis alignment construction), up to each axis's sign
    (eigenvectors are only defined up to sign).
    """
    # Distinct eigenvalues on each axis so eigh() can't mix them up.
    specs = [DroneGaussian(np.array([0.0, 0.0, 0.0]), np.diag([0.5**2, 0.3**2, 0.05**2]))]
    standard = StandardDroneRobot(urdf_path=None, gaussian_specs=specs)

    q = np.array([0.0, 0.0, 0.0, yaw])
    qdd = np.concatenate([a, [0.0]])

    cov = np.asarray(standard.collision_covariances_online(q, qdd), dtype=float).reshape(3, 3)
    assert np.allclose(cov, cov.T, atol=1e-8)  # still symmetric

    eigvals, eigvecs = np.linalg.eigh(cov)
    assert np.allclose(np.sort(eigvals), np.sort([0.05**2, 0.3**2, 0.5**2]), atol=1e-8)

    R_expected = _reference_rotation(a, yaw)

    # Body axis for the largest eigenvalue (0.5**2) is the body's local x.
    world_x_expected = R_expected[:, 0]
    world_x_actual = eigvecs[:, np.argmax(eigvals)]
    agreement = abs(float(np.dot(world_x_expected, world_x_actual)))
    assert agreement > 1 - 1e-6, f"expected {world_x_expected}, got ±{world_x_actual}"


def test_tilt_grows_monotonically_with_lateral_acceleration():
    """More lateral accel -> more of the long (local-y) axis swings into z."""
    specs = [DroneGaussian(np.array([0.0, 0.0, 0.0]), np.diag([0.05**2, 0.6**2, 0.05**2]))]
    standard = StandardDroneRobot(urdf_path=None, gaussian_specs=specs)
    q = np.array([0.0, 0.0, 0.0, 0.0])

    z_components = []
    for ay in [0.0, 1.0, 3.0, 6.0, 10.0]:
        qdd = np.array([0.0, ay, 0.0, 0.0])
        cov = np.asarray(standard.collision_covariances_online(q, qdd), dtype=float).reshape(3, 3)
        eigvals, eigvecs = np.linalg.eigh(cov)
        long_axis = eigvecs[:, np.argmax(eigvals)]
        z_components.append(abs(float(long_axis[2])))

    assert z_components[0] < 1e-9  # level flight: long axis purely horizontal
    assert all(b >= a - 1e-9 for a, b in zip(z_components, z_components[1:]))
    assert z_components[-1] > 0.3  # a real, not negligible, swing


# ---------------------------------------------------------------------------
# 2. The tilt reaches the planner's obstacle cost
# ---------------------------------------------------------------------------


def _manual_online_obstacle_cost(robot, environment, curve_np: np.ndarray, ddcurve_np: np.ndarray) -> float:
    """
    Evaluate problem_online.py's real obstacle-cost machinery at a
    hand-specified (q, qdd) trajectory -- no spline, no solve. Exercises the
    exact functions build_problem() calls.
    """
    num_samples, n_dof = curve_np.shape
    q_sym = cas.MX.sym("q", n_dof)
    n_gaussians = int(robot.collision_points(q_sym).shape[0])

    curve = cas.DM(curve_np)

    if robot.requires_acceleration:
        qdd_sym = cas.MX.sym("qdd", n_dof)
        ddcurve = cas.DM(ddcurve_np)
    else:
        qdd_sym = None
        ddcurve = None

    points_map = build_collision_point_map(
        robot, q_sym, n_gaussians, num_samples, curve, qdd_sym=qdd_sym, qdd_curve=ddcurve
    )
    covs_map = _build_collision_cov_map(
        robot, q_sym, n_gaussians, num_samples, curve, qdd_sym=qdd_sym, qdd_curve=ddcurve
    )

    obstacle_means_param = cas.DM(environment.build_per_sample_means(num_samples))
    obstacle_covs_param = cas.DM(environment.build_per_sample_covariances(num_samples))

    cost, _callbacks = _obstacle_cost(
        points_map, covs_map, obstacle_means_param, obstacle_covs_param,
        n_gaussians=n_gaussians, num_samples=num_samples, weight=1.0,
    )
    return float(cas.DM(cost))


def test_tilted_covariance_is_seen_by_the_online_obstacle_cost():
    """
    A flat (local-y elongated) Gaussian, level, has a small z-extent -- an
    obstacle placed just above it is invisible to the flat footprint. Bank
    it sideways (lateral qdd) and the long axis swings into z, reaching
    that same obstacle. The online NLP's obstacle cost must reflect this;
    a tilt-blind model (DroneRobot) with identical geometry must not.
    """
    specs = [DroneGaussian(np.array([0.0, 0.0, 0.0]), np.diag([0.06**2, 0.6**2, 0.06**2]))]

    # Just above the body's own flat extent (0.06 sigma-ish) but well within
    # reach of the tilted long axis (0.6) once it swings toward vertical.
    obstacle = StaticObstacle(np.array([0.0, 0.0, 0.35]), np.eye(3) * 0.03**2)
    environment = GaussianEnvironmentOnline(obstacles=[obstacle])

    num_samples = 1
    curve_np = np.array([[0.0, 0.0, 0.0, 0.0]])  # body at the origin, level
    ddcurve_level = np.zeros((num_samples, 4))
    ddcurve_banked = np.array([[0.0, 9.0, 0.0, 0.0]])  # hard lateral acceleration

    standard = StandardDroneRobot(urdf_path=None, gaussian_specs=specs)
    omni = DroneRobot(urdf_path=None, gaussian_specs=specs)

    cost_standard_level = _manual_online_obstacle_cost(standard, environment, curve_np, ddcurve_level)
    cost_standard_banked = _manual_online_obstacle_cost(standard, environment, curve_np, ddcurve_banked)
    cost_omni = _manual_online_obstacle_cost(omni, environment, curve_np, ddcurve_level)

    assert cost_standard_level < 1e-3
    assert cost_omni < 1e-3
    assert cost_standard_banked > 50 * max(cost_standard_level, cost_omni, 1e-6)
