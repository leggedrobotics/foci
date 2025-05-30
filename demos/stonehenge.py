import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize

import plyfile as ply

from foci.visualisation.vis_utils import ViserVis
from foci.utils.ply import extract_splat_data   

from foci.planners.planner import Planner

# Get local path
LOCAL = os.path.dirname(os.path.abspath(__file__))
ply_file = os.path.join(LOCAL, 'data/stonehenge.ply')

# ============================== Extract data from ply file ==============================
# read in ply file
means, covs, colors, opacities = extract_splat_data(ply_file)

means = 20*means
covs = 20**2 *covs

radius = max(np.linalg.norm(means, axis=1)) * 1.02

robot_cov = np.eye(3) * 0.01

planner = Planner(means, covs, robot_cov, num_control_points=10, num_samples=50) 




points = []
# add points at 1.2 * radius around the origin using cos and sin
for i in range(10):
    theta = i * 2 * np.pi / 10
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points.append([x, y, 0.5,np.pi/2])


#pick random start and end points from points array

start_point = points[3]
end_point = points[8]
opt_curve, spline  = planner.plan(start_point, end_point)

print(opt_curve)
print(spline)


# ============================== Visualize data ==============================
vis =ViserVis()
vis.add_gaussians(means, covs, color = colors,  opacity=opacities)
vis.add_points(np.array(points)[:,:3], color = [1,0,0])
vis.add_curve(spline[:,:3], color = [0,1,0])
vis.add_gaussian_path(opt_curve, robot_cov, planner.kinematics, color = [0,1,0])
vis.show()
