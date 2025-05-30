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

#scaling up
means = 20*means
covs = 20**2 *covs

radius = max(np.linalg.norm(means, axis=1)) * 1.02
robot_cov = np.eye(3) * 0.01

planner = Planner(means, covs, robot_cov, num_control_points=10, num_samples=40) 



points = []
for i in range(12):
    theta = i * 2 * np.pi / 12
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points.append([x, y, 0.5,np.pi/2])


solutions = []
for i in range(6):
    start_point = points[i]
    end_point = points[i+6]
    opt_curve, astar = planner.plan(start_point, end_point)
    solutions.append((opt_curve, astar))    



# ============================== Visualize data ==============================
vis =ViserVis()
vis.add_gaussians(means, covs, color = colors,  opacity=opacities)
vis.add_points(np.array(points)[:,:3], color = [1,0,0])
for i, (opt_curve, astar) in enumerate(solutions):
    vis.add_curve(astar[:,:3], color = [0,1,0], name = f"astar_{i}")
    vis.add_gaussian_path(opt_curve, robot_cov, planner.kinematics, color = [0,1,0], name = f"opt_curve_{i}")

vis.show()
