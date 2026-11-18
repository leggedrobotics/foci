import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data
from src.robots.manipulator import ManipulatorRobot, LinkGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance, path_length
from src.benchmark.stomp_planner import STOMPPlanner
from src.benchmark.chomp_planner import CHOMPPlanner


def benchmark_coffee():
    

    gaussian_specs = [
        LinkGaussian("base_link",          0.5, np.eye(3) * 0.05**2),
        LinkGaussian("base_link_inertia",  0.5, np.eye(3) * 0.05**2),
        LinkGaussian("shoulder_link",      0.3, np.eye(3) * 0.1**2),
        LinkGaussian("shoulder_link",      0.7, np.eye(3) * 0.1**2),
        LinkGaussian("upper_arm_link",     0.3, np.eye(3) * 0.2**2),
        LinkGaussian("upper_arm_link",     0.7, np.eye(3) * 0.2**2),
        LinkGaussian("forearm_link",       0.5, np.eye(3) * 0.2**2),
        LinkGaussian("wrist_1_link",       0.5, np.eye(3) * 0.1**2),
        LinkGaussian("wrist_2_link",       0.5, np.eye(3) * 0.1**2),
    ]
    


    theta_start = np.array([-np.pi, -2.4627256,  -1.0218003,  -2.5943978,   0.8556023,   0.41983527])
    theta_final = np.array([-0.76668125, -2.4627256,  -1.0218003,  -2.5943978,   0.8556023,   0.41983527])
    trajectory = np.array([theta_start, theta_start, theta_final])
# shoulder_pan=90°, shoulder_lift=-90° (vertical), elbow=-30° (mild fold), wrist_1..3=0

    goal = np.array([-1.5, 2.25, 0.5])

    robot = ManipulatorRobot(
        urdf_path="urdfs/ur5/ur5.urdf",
        root_link="base_link",
        tip_link="wrist_3_link",
        gaussian_specs=gaussian_specs,
    )

  

  

    vis = RobotVisualizer(
        robot=robot,
        trajectory=trajectory,
    )

    vis.visualize_goal(goal)
    vis.visualize_trajectory(loop=True)


if __name__ == "__main__":
    benchmark_coffee()
