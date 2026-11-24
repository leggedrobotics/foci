import os

import numpy as np

from src.utils.ply import extract_splat_data
from src.robots.manipulator import ManipulatorRobot, LinkGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance

RECORD_VIDEO = False
RECORD_FPS = 10

def coffe_table_demo():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    record_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "coffe_table.mp4"))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Coffee_Table.ply")

    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data(ply_file)

    translation = np.array([2.5, 0.0, 0.6])
    scale_factor = 0.005
    mesh_scale = 3.9
    
    floor_height_threshold = np.min(obstacle_means[:, 2]) + 0.1
    mask = obstacle_means[:, 2] > floor_height_threshold

    obstacle_means = obstacle_means[mask]
    obstacle_covs = obstacle_covs[mask]
    colors = colors[mask]
    opacities = opacities[mask]
    
    centroid = obstacle_means.mean(axis=0)
    obstacle_means = (obstacle_means - centroid) * mesh_scale + centroid + translation
    obstacle_covs = obstacle_covs * (scale_factor * mesh_scale)**2
     
    gaussian_specs = [
        LinkGaussian("base_link",          0.5, np.eye(3) * 0.1**2),
        LinkGaussian("base_link_inertia",  0.5, np.eye(3) * 0.1**2),
        LinkGaussian("shoulder_link",      0.3, np.eye(3) * 0.2**2),
        LinkGaussian("shoulder_link",      0.7, np.eye(3) * 0.2**2),
        LinkGaussian("upper_arm_link",     0.3, np.eye(3) * 0.2**2),
        LinkGaussian("upper_arm_link",     0.7, np.eye(3) * 0.2**2),
        LinkGaussian("forearm_link",       0.3, np.eye(3) * 0.2**2),
        LinkGaussian("forearm_link",       0.7, np.eye(3) * 0.2**2),
        LinkGaussian("wrist_1_link",       0.5, np.eye(3) * 0.1**2),
        LinkGaussian("wrist_2_link",       0.5, np.eye(3) * 0.1**2),
    ]

    theta_start = np.array([np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([1.4, 0.0, 0.75])

    robot = ManipulatorRobot(
        urdf_path="urdfs/ur5/ur5.urdf",
        root_link="base_link",
        tip_link="wrist_3_link",
        gaussian_specs=gaussian_specs,
    )


    joint_groups = JointGroups(
        virtual_indices=[], 
        virtual_wmax=4.0, 
        virtual_amax=3.5, 
        real_wmax=7.5, 
        real_amax=6.5,
    )
    
    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    initializer = RRTStarInitializer(
        voxel_size=0.00001,
        goal_threshold=0.001,
        random_seed=42,
        max_time=None,   
    )

    foci_planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
        num_control_points=12,
        num_samples=25,
        weights={
            "goal": 10.0,
            "obstacle": 50.0,
            "jerk": 0.0005,
            "virtual_jerk": 0.01,
        },
        vmax=8.0,
        linear_solver="ma27",
    )

    result = foci_planner.plan(
        start=theta_start,
        goal=goal,
    )
    

    min_dist, info = minimum_robot_environment_distance(robot,environment,result.trajectory)

    print("\n--- Planning timings ---")
    print(f"Initializer: {result.timings['initializer']:.6f} s")
    print(f"Build:       {result.timings['build']:.6f} s")
    print(f"Solver:      {result.timings['solve']:.6f} s")
    print(f"Total:       {result.timings['total']:.6f} s")
    print(f"Success:     {result.success}")

    print("\n--- Benchmark metrics ---")
    print(f"Number of environment gaussians: {len(obstacle_means)}")
    print(f"Number of robot gaussians: {len(gaussian_specs)}")
    print(f"Minimum robot-environment distance: {min_dist:.3f} m")

    vis = RobotVisualizer(
        robot=robot,
        trajectory=result.trajectory
    )
    
    vis.visualize_goal(goal)
    vis.visualize_gaussian_splat("Coffe table", obstacle_means, obstacle_covs, colors, opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_path()
    vis.visualize_initializer_path(result.initial_trajectory)
    vis.visualize_trajectory(loop=True, record=RECORD_VIDEO, record_path=record_path, record_fps=RECORD_FPS)
    

if __name__ == "__main__":
    coffe_table_demo()
