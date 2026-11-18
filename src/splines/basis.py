import numpy as np

"""
Basis matrices used for cubic B-spline evaluation and MINVO hull conversion.

Conventions
-----------
- BSPLINE_k stores the polynomial basis matrix for the k-th derivative order.
- MINVO_k stores the MINVO conversion matrix for the k-th derivative order.
- All matrices are expressed in power basis form.
"""

# ===================== MINVO MATRICES ======================
MINVO_3 = np.array(
    [
        [
            -3.4416309793565660335445954842726,
            6.9895482693324069156659561485867,
            -4.4622887879670974919932291413716,
            0.91437149799125659734933338484986,
        ],
        [
            6.6792587678886103930153694818728,
            -11.845989952130473454872117144987,
            5.2523596862506065630071816485724,
            -0.000000000000000055511151231257827021181583404541,
        ],
        [
            -6.6792587678886103930153694818728,
            8.1917863515353577241739913006313,
            -1.5981560856554908323090558042168,
            0.085628502008743445639282754200394,
        ],
        [
            3.4416309793565660335445954842726,
            -3.335344668737291184967830304231,
            0.80808518737198176129510329701588,
            -0.000000000000000012522535092207576212786079850048,
        ],
    ]
).T

MINVO_2 = np.array(
    [
        [
            1.4999999992328318931811281800037,
            -2.3660254034601951866889635311964,
            0.9330127021136816189983420599674,
        ],
        [-2.9999999984656637863622563600074, 2.9999999984656637863622563600074, 0],
        [
            1.4999999992328318931811281800037,
            -0.6339745950054685996732928288111,
            0.066987297886318325490506708774774,
        ],
    ]
).T

MINVO_1 = np.array(
    [
        [-1, 1],
        [1, 0],
    ]
).T

MINVO_0 = np.array(
    [
        [1],
    ]
).T

INV_MINVO_3 = np.linalg.inv(MINVO_3)
INV_MINVO_2 = np.linalg.inv(MINVO_2)
INV_MINVO_1 = np.linalg.inv(MINVO_1)
INV_MINVO_0 = np.linalg.inv(MINVO_0)


# ===================== B-SPLINE MATRICES ======================
BSPLINE_3 = np.array(
    [
        [
            -0.16666666666666666666666666666667,
            0.5,
            -0.5,
            0.16666666666666666666666666666667,
        ],
        [0.5, -1.0, 0, 0.66666666666666666666666666666667],
        [-0.5, 0.5, 0.5, 0.16666666666666666666666666666667],
        [0.16666666666666666666666666666667, 0, 0, 0],
    ]
).T

BSPLINE_2 = np.array([[0.5, -1.0, 0.5], [-1.0, 1.0, 0.5], [0.5, 0, 0]]).T

BSPLINE_1 = np.array([[-1, 1], [1, 0]]).T

BSPLINE_0 = np.array([[1]]).T


# ===================== BASIS FUNCTION ======================
def bspline_basis(t: float, derivative_order: int = 0):
    """
    Evaluate the cubic B-spline basis (or one of its derivatives) at a local
    parameter t in [0, 1].

    Parameters
    ----------
    t : float
        Local spline parameter inside a segment.
    derivative_order : int, default=0
        Derivative order to evaluate. Supported values are 0, 1, 2, and 3.

    Returns
    -------
    np.ndarray
        Basis weight vector of shape (4,).

    Raises
    ------
    ValueError
        If `derivative_order` is not in {0, 1, 2, 3}.
    """

    if derivative_order not in {0, 1, 2, 3}:
        raise ValueError(
            f"derivative_order must be one of {{0, 1, 2, 3}}, got {derivative_order}"
        )

    if derivative_order == 0:
        power_vector = np.array([t**3, t**2, t, 1.0])
    elif derivative_order == 1:
        power_vector = np.array([3.0 * t**2, 2.0 * t, 1.0, 0.0])
    elif derivative_order == 2:
        power_vector = np.array([6.0 * t, 2.0, 0.0, 0.0])
    else:
        power_vector = np.array([6.0, 0.0, 0.0, 0.0])

    return power_vector @ BSPLINE_3
