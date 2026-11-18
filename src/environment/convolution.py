
from __future__ import annotations

from abc import abstractmethod

import math
import warp as wp
import numpy as np
import casadi as cas

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
PI_CUBIC: float = math.sqrt((2.0 * math.pi) ** 3)
EPS = 1e-9

# Every kernel below reduces over a (num_points, num_obstacles) thread grid with
# wp.atomic_add, whose summation order is not reproducible.  The per-term maths
# stays in float32 (that is the precision of the uploaded splat data), but the
# ACCUMULATORS are float64.  Measured on 2e5 obstacles and 7 body points, the
# gradient's run-to-run spread at a FIXED x is 5.1e-6 relative with a float32
# accumulator and 1.5e-14 with a float64 one.  5e-6 of jitter on the gradient
# is enough to stall IPOPT's line search near its 1e-3 tolerance, since the
# solver assumes it is sampling one fixed function.
 
# ---------------------------------------------------------------------------
# Warp device-side helpers
# ---------------------------------------------------------------------------
 
@wp.func
def det33(m: wp.mat33) -> wp.float32:
    """Determinant of a 3�3 matrix."""
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[2, 0] * m[1, 2])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1])
    )
 
 
@wp.func
def inv33(m: wp.mat33) -> wp.mat33:
    """Inverse of a 3�3 matrix (assumes non-singular)."""
    inv_det = 1.0 / det33(m)
    inv = wp.mat33()
    inv[0, 0] = (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]) * inv_det
    inv[0, 1] = (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv_det
    inv[0, 2] = (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv_det
    inv[1, 0] = (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * inv_det
    inv[1, 1] = (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv_det
    inv[1, 2] = (m[1, 0] * m[0, 2] - m[0, 0] * m[1, 2]) * inv_det
    inv[2, 0] = (m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]) * inv_det
    inv[2, 1] = (m[2, 0] * m[0, 1] - m[0, 0] * m[2, 1]) * inv_det
    inv[2, 2] = (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) * inv_det
    return inv
 
 
@wp.func
def regularize33(m: wp.mat33, eps: wp.float32) -> wp.mat33:
    """Add eps to the diagonal of a 3�3 matrix for numerical stability."""
    out = m
    out[0, 0] = out[0, 0] + eps
    out[1, 1] = out[1, 1] + eps
    out[2, 2] = out[2, 2] + eps
    return out
 
 
@wp.func
def mat33_from_flat(flat: wp.array(dtype=wp.float32), offset: int) -> wp.mat33:
    """Read a row-major 3x3 matrix from a flat float32 array at `offset`."""
    return wp.mat33(
        flat[offset + 0], flat[offset + 1], flat[offset + 2],
        flat[offset + 3], flat[offset + 4], flat[offset + 5],
        flat[offset + 6], flat[offset + 7], flat[offset + 8],
    )
 
 
# ---------------------------------------------------------------------------
# Forward kernels
# ---------------------------------------------------------------------------
 
 
@wp.kernel
def kernel_forward_static(
    robot_means: wp.array(dtype=wp.vec3),
    obstacle_means: wp.array(dtype=wp.vec3),
    covs_det: wp.array(dtype=wp.float32),
    covs_inv: wp.array(dtype=wp.mat33),
    intermediate: wp.array(dtype=wp.float64),
    pi_cubic: wp.float32,
):
    """Forward pass for the static functor.
 
    The obstacle covariances are pre-convolved with the robot covariance and
    uploaded as `covs_det` / `covs_inv` at construction time.
 
    Thread grid: (num_points, num_obstacles).
    """
    m, n = wp.tid()
    diff = robot_means[m] - obstacle_means[n]
    pdf = (
        wp.exp(-0.5 * wp.dot(diff, covs_inv[n] @ diff))
        / (wp.sqrt(covs_det[n]) * pi_cubic)
    )
    wp.atomic_add(intermediate, n, wp.float64(pdf))
 
 
@wp.kernel
def kernel_forward_online(
    robot_means: wp.array(dtype=wp.vec3),
    robot_covs_flat: wp.array(dtype=wp.float32),
    obstacle_means: wp.array(dtype=wp.vec3),
    obstacle_covs_flat: wp.array(dtype=wp.float32),
    intermediate: wp.array(dtype=wp.float64),
    pi_cubic: wp.float32,
    eps: wp.float32,
):
    """Forward pass for the online functor.
 
    Covariances are passed at eval-time and summed on the fly.
 
    Thread grid: (num_points, num_obstacles).
    """
    m, n = wp.tid()
    diff = robot_means[m] - obstacle_means[n]
 
    cov_sum = regularize33(
        mat33_from_flat(robot_covs_flat, m * 9)
        + mat33_from_flat(obstacle_covs_flat, n * 9),
        eps,
    )
    pdf = wp.exp(-0.5 * wp.dot(diff, inv33(cov_sum) @ diff)) / (
        wp.sqrt(det33(cov_sum)) * pi_cubic
    )
    wp.atomic_add(intermediate, n, wp.float64(pdf))
 
 
# ---------------------------------------------------------------------------
# Jacobian kernels
# ---------------------------------------------------------------------------
 
 
@wp.kernel
def kernel_jacobian_static(
    robot_means: wp.array(dtype=wp.vec3),
    obstacle_means: wp.array(dtype=wp.vec3),
    covs_det: wp.array(dtype=wp.float32),
    covs_inv: wp.array(dtype=wp.mat33),
    grad_means: wp.array2d(dtype=wp.vec3),
    pi_cubic: wp.float32,
):
    """Jacobian of the forward cost w.r.t. robot_means for the static functor.

    Thread grid: (num_points, num_obstacles). Each thread owns a unique
    (m, n) slot in `grad_means` -- no atomics, no contention. The caller
    reduces over n (see _JacobianStatic.eval).

    This used to accumulate directly into a (num_points,) buffer with
    atomic_add to save memory, but for small num_points (CHOMP/STOMP always
    call this functor with num_points=1) that means every one of the
    num_obstacles threads races to add into the SAME memory location --
    measured ~19x slower than materialising the full (num_points,
    num_obstacles) buffer and reducing it with wp.utils.array_sum(axis=1).
    At float32, that buffer is num_points * num_obstacles * 12 bytes (~6.5 MB
    for num_points=1 and Bonsai's ~570k obstacles; ~160 MB at num_points=25),
    which is not a concern on this hardware. If num_points or num_obstacles
    grows enough for that to matter, prefer capping this buffer's size
    (e.g. chunking obstacles) over reintroducing the atomic accumulator.
    """
    m, n = wp.tid()
    diff = robot_means[m] - obstacle_means[n]
    inv = covs_inv[n]
    pdf = (
        wp.exp(-0.5 * wp.dot(diff, inv @ diff))
        / (wp.sqrt(covs_det[n]) * pi_cubic)
    )
    grad_means[m, n] = -pdf * inv @ diff
 
 
@wp.kernel
def kernel_jacobian_online(
    robot_means: wp.array(dtype=wp.vec3),
    robot_covs_flat: wp.array(dtype=wp.float32),
    obstacle_means: wp.array(dtype=wp.vec3),
    obstacle_covs_flat: wp.array(dtype=wp.float32),
    grad_means: wp.array(dtype=wp.vec3d),
    grad_robot_covs: wp.array(dtype=wp.float64),
    grad_obstacle_means: wp.array(dtype=wp.vec3d),
    grad_obstacle_covs: wp.array(dtype=wp.float64),
    pi_cubic: wp.float32,
    eps: wp.float32,
    normalizer: wp.float32,
):
    """Jacobian of the forward cost for the online functor.
 
    Differentiates w.r.t. robot means, robot covariances, obstacle means,
    and obstacle covariances.
 
    Thread grid: (num_points, num_obstacles).
    """
    m, n = wp.tid()
    diff = robot_means[m] - obstacle_means[n]
 
    cov_sum = regularize33(
        mat33_from_flat(robot_covs_flat, m * 9)
        + mat33_from_flat(obstacle_covs_flat, n * 9),
        eps,
    )
    inv = inv33(cov_sum)
    inv_diff = inv @ diff
    pdf = (
        wp.exp(-0.5 * wp.dot(diff, inv_diff))
        / (wp.sqrt(det33(cov_sum)) * pi_cubic)
    ) * normalizer
 
    # d cost / d mu_robot  =  -pdf * inv * diff
    grad_mu = -pdf * inv_diff
    grad_mu64 = wp.vec3d(
        wp.float64(grad_mu[0]), wp.float64(grad_mu[1]), wp.float64(grad_mu[2])
    )
    wp.atomic_add(grad_means, m, grad_mu64)
    wp.atomic_add(grad_obstacle_means, n, -grad_mu64)
 
    # d cost / d Sigma  =  0.5 * pdf * (inv_diff ? inv_diff - inv)
    robot_idx = m * 9
    obstacle_idx = n * 9
    for i in range(3):
        for j in range(3):
            g = wp.float64(0.5 * pdf * (inv_diff[i] * inv_diff[j] - inv[i, j]))
            wp.atomic_add(grad_robot_covs, robot_idx + i * 3 + j, g)
            wp.atomic_add(grad_obstacle_covs, obstacle_idx + i * 3 + j, g)

class BaseConvolutionFunctor(cas.Callback):
    """Abstract CasADi callback for Gaussian convolution costs.

    Subclasses must implement:
        get_n_in()
        get_sparsity_in(i)
        eval(arg)
        get_jacobian(name, inames, onames, opts)

    The output is always a scalar (the normalised mean pdf value).
    """

    # ------------------------------------------------------------------
    # Shared construction helpers
    # ------------------------------------------------------------------

    def _init_shared(self, num_points: int, num_obstacles: int) -> None:
        """Store dimensions and the normalisation constant."""
        self.num_points = int(num_points)
        self.num_obstacles = int(num_obstacles)
        self.pi_cubic = float(PI_CUBIC)
        self._normalizer = 1.0 / float(num_points * num_obstacles)

    # ------------------------------------------------------------------
    # CasADi Callback interface ? shared across all subclasses
    # ------------------------------------------------------------------

    def get_n_out(self) -> int:
        return 1

    def get_sparsity_out(self, i: int) -> cas.Sparsity:  # noqa: ARG002
        return cas.Sparsity.dense(1, 1)

    def has_jacobian(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_n_in(self) -> int: ...

    @abstractmethod
    def get_sparsity_in(self, i: int) -> cas.Sparsity: ...

    @abstractmethod
    def eval(self, arg): ...

    @abstractmethod
    def get_jacobian(self, name, inames, onames, opts): ...

class _JacobianStatic(cas.Callback):
    """Jacobian of ConvolutionFunctor w.r.t. robot body-point positions.
 
    Input  0: robot_means, shape (num_points, 3)
    Input  1: scalar cost output (passed by CasADi, unused here)
    Output 0: Jacobian row, shape (1, num_points * 3)
    """
 
    def __init__(
        self,
        name: str,
        num_points: int,
        obstacle_means_wp: wp.array,
        covs_det_wp: wp.array,
        covs_inv_wp: wp.array,
        normalizer: float,
        opts: dict,
    ) -> None:
        cas.Callback.__init__(self)
 
        self.num_points = num_points
        self.num_obstacles = len(obstacle_means_wp)
        self.obstacle_means = obstacle_means_wp
        self.covs_det = covs_det_wp
        self.covs_inv = covs_inv_wp
        self._normalizer = normalizer
        self.pi_cubic = float(PI_CUBIC)
 
        # (num_points, num_obstacles) buffer: one unique slot per thread, no
        # atomics. See the note on kernel_jacobian_static for why -- shrinking
        # this to (num_points,) with an atomic accumulator (the previous
        # approach) turns every one of the num_obstacles threads into a
        # contender on the same memory location whenever num_points is small,
        # which is CHOMP/STOMP's normal case (num_points=1) and measured
        # ~19x slower than this.
        self._grad_means = wp.zeros((num_points, self.num_obstacles), dtype=wp.vec3)
 
        self.construct(name, opts)
 
    def get_n_in(self) -> int:
        return 2  # robot_means + scalar output
 
    def get_n_out(self) -> int:
        return 1
 
    def get_sparsity_in(self, i: int) -> cas.Sparsity:
        if i == 0:
            return cas.Sparsity.dense(self.num_points, 3)
        return cas.Sparsity.dense(1, 1)  # scalar output
 
    def get_sparsity_out(self, i: int) -> cas.Sparsity:  # noqa: ARG002
        return cas.Sparsity.dense(1, self.num_points * 3)
 
    def eval(self, arg):
        robot_means = np.asarray(arg[0], dtype=np.float32).reshape(self.num_points, 3)
        robot_means_wp = wp.from_numpy(robot_means, dtype=wp.vec3)

        # Every (m, n) slot is overwritten (not accumulated into) by exactly
        # one thread, so no zero_() is needed before the launch.
        wp.launch(
            kernel=kernel_jacobian_static,
            dim=(self.num_points, self.num_obstacles),
            inputs=[
                robot_means_wp,
                self.obstacle_means,
                self.covs_det,
                self.covs_inv,
                self._grad_means,
                wp.float32(self.pi_cubic),
            ],
        )
 
        # GPU-side reduction over obstacles (axis=1), deterministic for a
        # fixed input buffer (see the note in ConvolutionFunctor.eval).
        # CasADi indexes a Jacobian row by vec(input), which is column-major,
        # hence the transpose of the (num_points, 3) buffer.
        grad = wp.utils.array_sum(self._grad_means, axis=1)  # (num_points,) vec3
        out = grad.numpy().T.reshape(1, self.num_points * 3)
        return [(out * self._normalizer).astype(np.float64)]
 
 
class ConvolutionFunctor(BaseConvolutionFunctor):
    """Static Gaussian convolution functor.
 
    Obstacle statistics are fixed at construction; only robot body-point
    positions vary during optimisation.
 
    Parameters
    ----------
    name:
        CasADi callback name.
    num_points:
        Number of robot body points.
    obstacle_means:
        Shape (n_obstacles, 3).
    covs_det:
        Shape (n_obstacles,). Determinants of (obstacle + robot) covariances.
    covs_inv:
        Shape (n_obstacles, 3, 3). Inverses of (obstacle + robot) covariances.
    opts:
        CasADi callback options.
    """
 
    def __init__(
        self,
        name: str,
        num_points: int,
        obstacle_means: np.ndarray,
        covs_det: np.ndarray,
        covs_inv: np.ndarray,
        opts: dict | None = None,
    ) -> None:
        cas.Callback.__init__(self)
 
        if opts is None:
            opts = {}
 
        num_obstacles = len(obstacle_means)
        self._init_shared(num_points, num_obstacles)
 
        # Upload fixed obstacle data to the Warp device once
        self._obstacle_means = wp.from_numpy(
            obstacle_means.astype(np.float32), dtype=wp.vec3
        )
        self._covs_det = wp.from_numpy(
            covs_det.astype(np.float32), dtype=wp.float32
        )
        self._covs_inv = wp.from_numpy(
            covs_inv.astype(np.float32), dtype=wp.mat33
        )
 
        self._intermediate = wp.zeros(num_obstacles, dtype=wp.float64)
        self._jacobian_callback: _JacobianStatic | None = None
        self._opts = opts
 
        self.construct(name, opts)
 
    # ------------------------------------------------------------------
    # CasADi Callback interface
    # ------------------------------------------------------------------
 
    def get_n_in(self) -> int:
        return 1  # robot_means only
 
    def get_sparsity_in(self, i: int) -> cas.Sparsity:
        if i == 0:
            return cas.Sparsity.dense(self.num_points, 3)
        return cas.Sparsity.dense(0, 0)
 
    def eval(self, arg):
        robot_means = np.asarray(arg[0], dtype=np.float32).reshape(self.num_points, 3)
        robot_means_wp = wp.from_numpy(robot_means, dtype=wp.vec3)
 
        self._intermediate.zero_()
        wp.launch(
            kernel=kernel_forward_static,
            dim=(self.num_points, self.num_obstacles),
            inputs=[
                robot_means_wp,
                self._obstacle_means,
                self._covs_det,
                self._covs_inv,
                self._intermediate,
                wp.float32(self.pi_cubic),
            ],
        )
 
        # GPU-side reduction: pulling the full (num_obstacles,) buffer to the
        # host before summing (self._intermediate.numpy().sum()) measured 22x
        # slower here for Bonsai's ~570k obstacles -- it is what made CHOMP's
        # per-waypoint, per-body-point obstacle evaluation (called far more
        # often than FOCI's, since it has no batching across waypoints) ~6x
        # slower after the float32->float64 accumulator change below.
        # wp.utils.array_sum is deterministic given a fixed input buffer
        # (verified bit-identical over 30 repeated calls); the atomic_add
        # order inside the kernel above is the actual source of run-to-run
        # jitter the float64 accumulator fixes, and this reduction step runs
        # after that buffer is already finalized, so it adds none of its own.
        total = float(wp.utils.array_sum(self._intermediate))
        return [np.array([[total * self._normalizer]], dtype=np.float64)]
 
    def get_jacobian(self, name, inames, onames, opts):
        if self._jacobian_callback is None:
            self._jacobian_callback = _JacobianStatic(
                name,
                self.num_points,
                self._obstacle_means,
                self._covs_det,
                self._covs_inv,
                self._normalizer,
                opts,
            )
        return self._jacobian_callback


class _JacobianOnline(cas.Callback):
    """Jacobian of ConvolutionFunctorOnline w.r.t. all four inputs.

    Inputs  (5 total = 4 original inputs + scalar output passed by CasADi):
        0. robot_means       (num_points,  3)
        1. robot_covs_flat   (num_points,  9)
        2. obstacle_means    (num_obstacles, 3)
        3. obstacle_covs_flat(num_obstacles, 9)
        4. scalar cost output (unused)

    Outputs (4):
        0. d cost / d robot_means       shape (1, num_points  * 3)
        1. d cost / d robot_covs_flat   shape (1, num_points  * 9)
        2. d cost / d obstacle_means    shape (1, num_obstacles * 3)
        3. d cost / d obstacle_covs_flat shape (1, num_obstacles * 9)

    Returning all four blocks is required because robot covariances typically
    depend on q through the forward kinematics uncertainty propagation.
    """

    def __init__(
        self,
        name: str,
        num_points: int,
        num_obstacles: int,
        normalizer: float,
        opts: dict,
    ) -> None:
        cas.Callback.__init__(self)

        self.num_points = int(num_points)
        self.num_obstacles = int(num_obstacles)
        self._normalizer = normalizer
        self.pi_cubic = float(PI_CUBIC)

        # Gradient buffers; allocated lazily and reused across calls
        self._grad_means: wp.array | None = None
        self._grad_robot_covs: wp.array | None = None
        self._grad_obstacle_means: wp.array | None = None
        self._grad_obstacle_covs: wp.array | None = None

        self.construct(name, opts)

    # ------------------------------------------------------------------
    # CasADi Callback interface
    # ------------------------------------------------------------------

    def get_n_in(self) -> int:
        return 5

    def get_n_out(self) -> int:
        return 4

    def get_sparsity_in(self, i: int) -> cas.Sparsity:
        if i == 0:
            return cas.Sparsity.dense(self.num_points, 3)
        if i == 1:
            return cas.Sparsity.dense(self.num_points, 9)
        if i == 2:
            return cas.Sparsity.dense(self.num_obstacles, 3)
        if i == 3:
            return cas.Sparsity.dense(self.num_obstacles, 9)
        return cas.Sparsity.dense(1, 1)  # scalar output (input 4)

    def get_sparsity_out(self, i: int) -> cas.Sparsity:
        if i == 0:
            return cas.Sparsity.dense(1, self.num_points * 3)
        if i == 1:
            return cas.Sparsity.dense(1, self.num_points * 9)
        if i == 2:
            return cas.Sparsity.dense(1, self.num_obstacles * 3)
        if i == 3:
            return cas.Sparsity.dense(1, self.num_obstacles * 9)
        return cas.Sparsity.dense(0, 0)

    def eval(self, arg):
        robot_means = np.asarray(arg[0], dtype=np.float32).reshape(self.num_points, 3)
        robot_covs = np.asarray(arg[1], dtype=np.float32).reshape(self.num_points, 9)
        obstacle_means = np.asarray(arg[2], dtype=np.float32).reshape(self.num_obstacles, 3)
        obstacle_covs = np.asarray(arg[3], dtype=np.float32).reshape(self.num_obstacles, 9)

        self._ensure_buffers()

        robot_means_wp = wp.from_numpy(robot_means, dtype=wp.vec3)
        robot_covs_wp = wp.from_numpy(robot_covs.reshape(-1), dtype=wp.float32)
        obstacle_means_wp = wp.from_numpy(obstacle_means, dtype=wp.vec3)
        obstacle_covs_wp = wp.from_numpy(obstacle_covs.reshape(-1), dtype=wp.float32)

        self._grad_means.zero_()
        self._grad_robot_covs.zero_()
        self._grad_obstacle_means.zero_()
        self._grad_obstacle_covs.zero_()

        wp.launch(
            kernel=kernel_jacobian_online,
            dim=(self.num_points, self.num_obstacles),
            inputs=[
                robot_means_wp,
                robot_covs_wp,
                obstacle_means_wp,
                obstacle_covs_wp,
                self._grad_means,
                self._grad_robot_covs,
                self._grad_obstacle_means,
                self._grad_obstacle_covs,
                wp.float32(self.pi_cubic),
                wp.float32(EPS),
                wp.float32(self._normalizer),
            ],
        )

        # CasADi indexes a Jacobian row by vec(input), which is COLUMN-major.
        # The vec3 buffers give (N, 3) and can be transposed directly; the
        # covariance buffers are flat (N * 9,), so a bare .transpose() is a
        # no-op on them and they need the intermediate reshape.
        n_p, n_o = self.num_points, self.num_obstacles

        return [
            self._grad_means.numpy().T.reshape(1, n_p * 3).astype(np.float64),
            self._grad_robot_covs.numpy().reshape(n_p, 9).T.reshape(1, n_p * 9).astype(np.float64),
            self._grad_obstacle_means.numpy().T.reshape(1, n_o * 3).astype(np.float64),
            self._grad_obstacle_covs.numpy().reshape(n_o, 9).T.reshape(1, n_o * 9).astype(np.float64),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_buffers(self) -> None:
        """Allocate or reallocate gradient buffers if dimensions changed."""
        if self._grad_means is None or self._grad_means.shape[0] != self.num_points:
            self._grad_means = wp.zeros(self.num_points, dtype=wp.vec3d)
            self._grad_robot_covs = wp.zeros(self.num_points * 9, dtype=wp.float64)

        if (
            self._grad_obstacle_means is None
            or self._grad_obstacle_means.shape[0] != self.num_obstacles
        ):
            self._grad_obstacle_means = wp.zeros(self.num_obstacles, dtype=wp.vec3d)
            self._grad_obstacle_covs = wp.zeros(self.num_obstacles * 9, dtype=wp.float64)


class ConvolutionFunctorOnline(BaseConvolutionFunctor):
    """Online Gaussian convolution functor.

    All four tensors (robot means, robot covariances, obstacle means, obstacle
    covariances) are passed at eval-time, making this suitable for problems
    where robot or obstacle covariances depend on optimisation variables.

    Parameters
    ----------
    name:
        CasADi callback name.
    num_points:
        Number of robot body points.
    num_obstacles:
        Number of obstacles.
    opts:
        CasADi callback options.
    """

    def __init__(
        self,
        name: str,
        num_points: int,
        num_obstacles: int,
        opts: dict | None = None,
    ) -> None:
        cas.Callback.__init__(self)

        if opts is None:
            opts = {}

        self._init_shared(num_points, num_obstacles)

        self._intermediate: wp.array | None = None
        self._jacobian_callback: _JacobianOnline | None = None
        self._opts = opts

        self.construct(name, opts)

    # ------------------------------------------------------------------
    # CasADi Callback interface
    # ------------------------------------------------------------------

    def get_n_in(self) -> int:
        return 4

    def get_sparsity_in(self, i: int) -> cas.Sparsity:
        if i == 0:
            return cas.Sparsity.dense(self.num_points, 3)
        if i == 1:
            return cas.Sparsity.dense(self.num_points, 9)
        if i == 2:
            return cas.Sparsity.dense(self.num_obstacles, 3)
        if i == 3:
            return cas.Sparsity.dense(self.num_obstacles, 9)
        return cas.Sparsity.dense(0, 0)

    def eval(self, arg):
        robot_means = np.asarray(arg[0], dtype=np.float32).reshape(self.num_points, 3)
        robot_covs = np.asarray(arg[1], dtype=np.float32).reshape(self.num_points, 9)
        obstacle_means = np.asarray(arg[2], dtype=np.float32).reshape(self.num_obstacles, 3)
        obstacle_covs = np.asarray(arg[3], dtype=np.float32).reshape(self.num_obstacles, 9)

        self._ensure_intermediate()

        robot_means_wp = wp.from_numpy(robot_means, dtype=wp.vec3)
        robot_covs_wp = wp.from_numpy(robot_covs.reshape(-1), dtype=wp.float32)
        obstacle_means_wp = wp.from_numpy(obstacle_means, dtype=wp.vec3)
        obstacle_covs_wp = wp.from_numpy(obstacle_covs.reshape(-1), dtype=wp.float32)

        self._intermediate.zero_()
        wp.launch(
            kernel=kernel_forward_online,
            dim=(self.num_points, self.num_obstacles),
            inputs=[
                robot_means_wp,
                robot_covs_wp,
                obstacle_means_wp,
                obstacle_covs_wp,
                self._intermediate,
                wp.float32(self.pi_cubic),
                wp.float32(EPS),
            ],
        )

        # See the note in ConvolutionFunctor.eval: GPU-side reduction, not a
        # full host copy of the (num_obstacles,) buffer.
        total = float(wp.utils.array_sum(self._intermediate))
        return [np.array([[total * self._normalizer]], dtype=np.float64)]

    def get_jacobian(self, name, inames, onames, opts):
        if self._jacobian_callback is None:
            self._jacobian_callback = _JacobianOnline(
                name,
                self.num_points,
                self.num_obstacles,
                self._normalizer,
                opts,
            )
        return self._jacobian_callback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_intermediate(self) -> None:
        if (
            self._intermediate is None
            or self._intermediate.shape[0] != self.num_obstacles
        ):
            self._intermediate = wp.zeros(self.num_obstacles, dtype=wp.float64)