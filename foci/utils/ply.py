
import plyfile as ply
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial.transform import Rotation as R


def extract_splat_data(ply_file):
    """Extracts the splat data from a ply file.
    Args:
        ply_file (str): Path to the ply file.
    Returns:
        means (np.ndarray): Means of the splats.
        covs (np.ndarray): Covariances of the splats.
        opacities (np.ndarray): Opacities of the splats.
    """
    # read in ply file
    plydata = ply.PlyData.read(ply_file)
    print(len(plydata['vertex']))
    splat = plydata['vertex']   

    SH_C0 = 0.28209479177387814

    means = structured_to_unstructured(splat[["x", "y", "z"]])
    scales = np.exp(structured_to_unstructured(splat[['scale_0', 'scale_1', 'scale_2']]))
    wxyzs = structured_to_unstructured(splat[['rot_0', 'rot_1', 'rot_2', 'rot_3']])
    colors = 0.5 + SH_C0 * structured_to_unstructured(splat[['f_dc_0', 'f_dc_1', 'f_dc_2']])
    opacities = 1.0 / (1.0 + np.exp(-splat["opacity"][:, None]))

    Rs = np.zeros((len(splat), 3, 3))
    for i in range(len(splat)):
        Rs[i] = R.from_quat([wxyzs[i][1],
                             wxyzs[i][2],
                             wxyzs[i][3],
                             wxyzs[i][0]]).as_matrix()

    # recontruct covariances from scales and quat
    covs = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )

    return means, covs, colors, opacities