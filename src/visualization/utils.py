import numpy as np
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass

@dataclass(frozen=True)
class EllipsoidFactory:
    """
    Convert covariance matrices into ellipsoid radii and orientations.

    Parameters
    ----------
    n_std : float, default=2.0
        Number of standard deviations used to define the ellipsoid radius.
    """

    n_std: float = 2.0

    def cov_to_ellipsoid(self, cov: np.ndarray):
        """
        Convert a covariance matrix into ellipsoid geometry.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix of shape `(3, 3)`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `(radii, quat_wxyz)` where:
            - `radii` has shape `(3,)`
            - `quat_wxyz` is the ellipsoid orientation quaternion in WXYZ order
        """
        eigvals, eigvecs = np.linalg.eigh(cov)

        radii = self.n_std * np.sqrt(np.maximum(eigvals, 1e-12))

        rotation_matrix = eigvecs
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] *= -1

        quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return radii, quat_wxyz