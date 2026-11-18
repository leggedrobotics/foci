import numpy as np
import casadi as cas

from src.splines.basis import bspline_basis


class BSpline:
    """
    Uniform cubic B-spline defined by a set of control points.

    Parameters
    ----------
    control_points : cas.MX
        CasADi matrix of shape (num_control_points, dim), where each row is a control point.
    """

    def __init__(self, control_points: cas.MX):
        self.control_points = control_points
        self.n_control_points = control_points.shape[0]
        self.dimension = control_points.shape[1]
        self.max_parameter = self.n_control_points - 3

    def _build_basis_matrix(
        self,
        sample_parameters: np.ndarray,
        derivative_order: int = 0,
    ) -> np.ndarray:
        """
        Build the global basis matrix for a set of sample parameters.

        Parameters
        ----------
        sample_parameters : np.ndarray
            1D array of parameter values in [0, max_parameter].
        derivative_order : int, default=0
            Derivative order of the spline basis.

        Returns
        -------
        np.ndarray
            Basis matrix of shape (num_samples, num_control_points).
        """

        basis_matrix = np.zeros((sample_parameters.shape[0], self.n_control_points))

        for i, t in enumerate(sample_parameters):
            segment_idx = max(min(int(np.floor(t)), self.max_parameter - 1), 0)
            local_t = t - segment_idx
            basis_matrix[i, segment_idx : segment_idx + 4] = bspline_basis(
                local_t,
                derivative_order=derivative_order,
            )

        return basis_matrix
    
    def sample_parameters(self, num_samples: int) -> np.ndarray:
        """
        Uniformly spaced parameters covering the full spline domain.

        Returns
        -------
        np.ndarray
            1D array of `num_samples` values spanning [0, max_parameter].
        """
        return np.linspace(0.0, self.max_parameter, num_samples)
    
    def basis_matrix(
        self,
        sample_parameters: np.ndarray,
        derivative_order: int = 0,
    ) -> np.ndarray:
        """
        Basis matrix evaluated at arbitrary parameter values.

        Useful to fit control points to a given set of samples without going
        through the control points stored in this instance.

        Parameters
        ----------
        sample_parameters : array-like
            Parameter values in [0, max_parameter].
        derivative_order : int, default=0
            Derivative order of the spline basis.

        Returns
        -------
        np.ndarray
            Basis matrix of shape (len(sample_parameters), num_control_points).
        """
        return self._build_basis_matrix(
            np.atleast_1d(np.asarray(sample_parameters, dtype=float)),
            derivative_order=derivative_order,
        )

    def spline_eval(self, num_samples: int, derivative_order: int = 0) -> np.ndarray:
        """
        Evaluate the spline or one of its derivatives.

        Parameters
        ----------
        num_samples : int
            Number of samples along the full spline domain.
        derivative_order : int, default=0
            Derivative order to evaluate.

        Returns
        -------
        np.ndarray
            Evaluated points with shape (num_samples, dim).
        """
        sample_parameters = np.linspace(0.0, self.max_parameter, num_samples)
        basis_matrix = self._build_basis_matrix(
            sample_parameters,
            derivative_order=derivative_order,
        )
        return basis_matrix @ self.control_points
