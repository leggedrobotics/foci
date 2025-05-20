import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize

import plyfile as ply

from foci.visualisation.vis_utils import ViserVis
from foci.utils.ply import extract_splat_data   


# Get local path
LOCAL = os.path.dirname(os.path.abspath(__file__))
ply_file = os.path.join(LOCAL, 'data/stonehenge.ply')

# ============================== Extract data from ply file ==============================
# read in ply file
means, covs, colors, opacities = extract_splat_data(ply_file)   


# ============================== Visualize data ==============================
vis =ViserVis()
vis.add_gaussians(means, covs, opacity=opacities)
vis.show()
