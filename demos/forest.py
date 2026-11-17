import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data_2
from src.robots.drone import DroneRobot, DroneGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance

RECORD_VIDEO = False
RECORD_FPS = 10

def forest_demo():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    record_path = os.path.join(os.path.dirname(__file__), "forest.mp4")
    ply_file = os.path.join(PROJECT_ROOT, "data/Forest.ply")

    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)

    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    translation = np.array([8.0, 0.0, 0.0])
    scale_factor = 10.0
    
    
    obstacle_means = (obstacle_means * scale_factor) @ rotation.T + translation
    obstacle_covs = np.einsum("ij,njk,lk->nil", rotation, obstacle_covs, rotation) * scale_factor**2

    

    gaussian_specs = [DroneGaussian(np.array([0.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.1 ** 2)]

    theta_start = np.array([0.0, -3.0, 1.0, 0.0])
    goal = np.array([14.0, 5.0, 3.5])

    robot = DroneRobot(
        urdf_path="urdfs/drone_example.urdf",
        arm_length=0.15,
        gaussian_specs=gaussian_specs,
    )

    joint_groups = JointGroups(
        virtual_indices=[0,1,2], 
        virtual_wmax=5.0, 
        virtual_amax=3.0, 
        real_wmax=5.0, 
        real_amax=3.0,
    )
    
    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    initializer = RRTStarInitializer(
        voxel_size=0.01,
        goal_threshold=0.005,
        random_seed=42,
        max_time=None,   
    )

    planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        #initializer=initializer,
        num_control_points=12,
        num_samples=25,
        weights={
            "goal": 100.0,
            "obstacle": 200.0,
            "jerk": 0.1,
            "virtual_jerk": 0.001,
        },
        vmax=1.0,
        linear_solver="ma27"
    )

    result = planner.plan(
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
        trajectory=result.trajectory,
        #follow_camera=True,
        #camera_offset=np.array([0.0, 0.0, 0.1]),
    )
    
    vis.visualize_goal(goal)
    vis.visualize_gaussian_splat("Coffe table", obstacle_means, obstacle_covs, colors, opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_path()
    vis.visualize_initializer_path(result.initial_trajectory)
    vis.visualize_trajectory(loop=True, record=RECORD_VIDEO, record_path=record_path, record_fps=RECORD_FPS)
    

if __name__ == "__main__":
    forest_demo()
