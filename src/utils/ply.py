import plyfile as ply
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial.transform import Rotation as R


def extract_splat_data(ply_file):

    plydata = ply.PlyData.read(ply_file)
    splat = plydata['vertex']
    
    # np.vstack(...).T is a transpose, so it's F-contiguous, not C-contiguous.
    # viser's add_gaussian_splats reinterprets centers' raw bytes
    # (.view(np.uint8)), which requires a C-contiguous last axis.
    means = np.ascontiguousarray(np.vstack([splat['x'], splat['y'], splat['z']]).T)
    covs = np.array([np.eye(3) for _ in range(len(splat))])
    colors = np.ascontiguousarray(np.vstack([splat['red'], splat['green'], splat['blue']]).T / 255.0)
    opacities = np.ones((len(splat), 1))

    return means, covs, colors, opacities


def extract_splat_data_2(ply_file):
    plydata = ply.PlyData.read(ply_file)
    splat = plydata["vertex"]


    means = structured_to_unstructured(splat[["x", "y", "z"]])

    SH_C0 = 0.28209479177387814
    colors = 0.5 + SH_C0 * structured_to_unstructured(
        splat[["f_dc_0", "f_dc_1", "f_dc_2"]]
    )
    opacities = 1.0 / (1.0 + np.exp(-splat["opacity"][:, None]))

    scales = np.exp(
        structured_to_unstructured(splat[["scale_0", "scale_1", "scale_2"]])
    )

    wxyzs = structured_to_unstructured(
        splat[["rot_0", "rot_1", "rot_2", "rot_3"]]
    )

    Rs = np.zeros((len(splat), 3, 3))

    for i in range(len(splat)):
        Rs[i] = R.from_quat(
            [wxyzs[i][1], wxyzs[i][2], wxyzs[i][3], wxyzs[i][0]]
        ).as_matrix()

    covs = np.einsum(
        "nij,njk,nlk->nil",
        Rs,
        np.eye(3)[None, :, :] * scales[:, None, :] ** 2,
        Rs,
    )

    return means, covs, colors, opacities
