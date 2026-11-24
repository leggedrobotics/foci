import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data_2
from src.robots.manipulator import ManipulatorRobot, LinkGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer, RobotVisualizerOnline
from src.benchmark.utils import (
    median_solve_time,
    minimum_robot_environment_distance,
    orbit_camera_params,
    planner_metrics,
    save_results_csv,
    task_path_length,
    scene_size,
)
from src.benchmark.stomp_planner import STOMPPlanner
from src.benchmark.chomp_planner import CHOMPPlanner

# RECORD_VIDEO = True records two videos with the camera orbiting the scene
# once each (ORBIT_PERIOD seconds) in docs/: benchmark_bonsai_foci.mp4 with the FOCI
# trajectory only, then benchmark_bonsai_all.mp4 with FOCI, RRT*, CHOMP and
# STOMP. It needs a browser connected to the viser URL; each recording stops
# by itself after one orbit. With False, the interactive viewer shows all of
# them. Every run also writes its statistics and final trajectories to
# docs/results.csv (see save_results_csv), replacing this scene's previous rows.
RECORD_VIDEO = False
RECORD_FPS = 10
ORBIT_PERIOD = 26.0


def benchmark_bonsai():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    record_path_foci = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "benchmark_bonsai_foci.mp4"))
    record_path_all = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "benchmark_bonsai_all.mp4"))
    results_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "results.csv"))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Bonsai.ply")
    
    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)

    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    translation = np.array([0.0, 2.2, 1.0])
    scale_factor = 3.0
    
    obstacle_means = (obstacle_means * scale_factor) @ rotation.T + translation
    obstacle_covs = np.einsum("ij,njk,lk->nil", rotation, obstacle_covs, rotation) * scale_factor**2
    
    obstacle_means = np.ascontiguousarray(obstacle_means, dtype=np.float32)
    obstacle_covs = np.ascontiguousarray(obstacle_covs, dtype=np.float32)


    gaussian_specs = [
        LinkGaussian("shoulder_link",   0.5, np.eye(3) * 0.1**2),
        LinkGaussian("upper_arm_link",  0.3, np.eye(3) * 0.2**2),
        LinkGaussian("upper_arm_link",  0.7, np.eye(3) * 0.2**2),  
        LinkGaussian("forearm_link",    0.5, np.eye(3) * 0.2**2),
        LinkGaussian("wrist_1_link",    0.5, np.eye(3) * 0.2**2),
        LinkGaussian("wrist_2_link",    0.5, np.eye(3) * 0.1**2),
    ]

    #theta_start = np.array([1.05, -0.23, -1.6, 1.21, -0.85, 0.02])
    theta_start = np.array([-3*np.pi/4,-1.8,  -1.0218003,  -2.5943978,   0.8556023,   0.41983527])
    
    goal = np.array([-1.5, 2.25, 0.5])

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
        real_wmax=5.5, 
        real_amax=4.0,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    initializer = RRTStarInitializer(
        voxel_size=0.01,
        goal_threshold=0.5,
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
            "goal": 2.0,
            "obstacle": 16.0,
            "jerk":0.05,
            "virtual_jerk": 0.01,
        },
        vmax=1.5,
        linear_solver="ma27",
    )

    
    chomp_planner = CHOMPPlanner(
        robot=robot, 
        environment=environment, 
        joint_groups=joint_groups, 
        num_waypoints=25,
        max_iter=1_000, 
        learning_rate=0.01, 
        weights={"obstacle": 1.0, "smoothness": 1.0},
        convergence_tol=1e-3,            
    )
    
    stomp_planner = STOMPPlanner(
        robot=robot, 
        environment=environment, 
        joint_groups=joint_groups, 
        num_waypoints=12, 
        n_samples=25, 
        max_iter=1_000,
        temperature=10, 
        weights={"obstacle": 1.0, "jerk": 1.0, "constraint": 1.0},
        noise_scale=0.1,
        convergence_tol=1e-3, 
        total_time=2.0,
    )
    
 
  
    # FOCI only: 1 untimed warmup + median of 3 timed calls, matching the
    # FOCI paper's eval/baselines.py methodology (_median_time) so this
    # number stays comparable with the earlier experiments. CHOMP/STOMP are
    # called once: both build their CasADi/warp machinery once in __init__
    # (outside this call), and since FOCI already ran first in this same
    # process, warp's kernel cache (shared across ConvolutionFunctor
    # instances, process-wide) is already warm by the time these run -- a
    # warmup here would just re-pay their solve for no JIT-avoidance
    # benefit, only extra wall time. (Same treatment as benchmark_coffe.py
    # and benchmark_forest.py -- this file had been left on the old,
    # inconsistent all-three-median methodology.)
    foci_result, foci_wall = median_solve_time(
        lambda: foci_planner.plan(start=theta_start, goal=goal)
    )

    init_trajectory = np.ascontiguousarray(foci_result.initial_trajectory, dtype=np.float32)

    theta_final = np.array(foci_result.trajectory[-1], dtype=np.float32, copy=True)

    chomp_result = chomp_planner.plan(
        start=theta_start, goal=theta_final, initial_trajectory=init_trajectory,
    )

    stomp_result = stomp_planner.plan(
        start=theta_start, goal=theta_final, initial_trajectory=foci_result.initial_trajectory,
    )

    # opacities=opacities drops near-transparent splats before measuring
    # clearance (opacity <= 0.5), matching the paper eval's OPACITY_MIN.
    foci_min_dist, info = minimum_robot_environment_distance(
        robot, environment, foci_result.trajectory, opacities=opacities, include_robot_covariance=False
    )
    chomp_min_dist, info = minimum_robot_environment_distance(
        robot, environment, chomp_result.trajectory, opacities=opacities, include_robot_covariance=False
    )
    stomp_min_dist, info = minimum_robot_environment_distance(
        robot, environment, stomp_result.trajectory, opacities=opacities, include_robot_covariance=False
    )

    foci_path_length = task_path_length(robot, foci_result.trajectory)
    chomp_path_length = task_path_length(robot, chomp_result.trajectory)
    stomp_path_length = task_path_length(robot, stomp_result.trajectory)

    # Dimensionless, comparable across scenes of different physical size
    # (the paper eval reports length / scene_size, not a raw metre figure).
    scene = scene_size(obstacle_means)

    print("\n--- Planning timings FOCI---")
    print(f"Median solve (3 reps, 1 warmup): {foci_wall:.6f} s")
    print(f"Last call -- Initial guess: {foci_result.timings['initializer']:.6f} s")
    print(f"Last call -- Build:  {foci_result.timings['build']:.6f} s")
    print(f"Last call -- Solver: {foci_result.timings['solve']:.6f} s")
    print(f"Last call -- Total:  {foci_result.timings['total']:.6f} s")
    print(f"Success:     {foci_result.success}")



    print("\n--- Planning timings CHOMP---")
    print(f"Initial guess: {chomp_result.timings['initializer']:.6f} s")
    print(f"Build:  {chomp_result.timings['build']:.6f} s")
    print(f"Solver: {chomp_result.timings['solve']:.6f} s")
    print(f"Total:  {chomp_result.timings['total']:.6f} s")
    print(f"Success:     {chomp_result.success}")

    print("\n--- Planning timings STOMP---")
    print(f"Initial guess: {stomp_result.timings['initializer']:.6f} s")
    print(f"Build:  {stomp_result.timings['build']:.6f} s")
    print(f"Solver: {stomp_result.timings['solve']:.6f} s")
    print(f"Total:  {stomp_result.timings['total']:.6f} s")
    print(f"Success:     {stomp_result.success}")

    print("\n--- Benchmark metrics ---")
    print(f"Number of environment gaussians: {len(obstacle_means)}")
    print(f"Number of robot gaussians: {len(gaussian_specs)}")
    print(f"Minimum robot-environment distance FOCI: {foci_min_dist:.3f} m")
    print(f"Minimum robot-environment distance CHOMP: {chomp_min_dist:.3f} m")
    print(f"Minimum robot-environment distance STOMP: {stomp_min_dist:.3f} m")
    print(f"Path length FOCI:  {foci_path_length:.3f} m  ({foci_path_length / scene:.3f} x scene diagonal)")
    print(f"Path length CHOMP: {chomp_path_length:.3f} m  ({chomp_path_length / scene:.3f} x scene diagonal)")
    print(f"Path length STOMP: {stomp_path_length:.3f} m  ({stomp_path_length / scene:.3f} x scene diagonal)")



    rrt_min_dist, info = minimum_robot_environment_distance(
        robot, environment, foci_result.initial_trajectory, opacities=opacities, include_robot_covariance=False
    )
    rrt_path_length = task_path_length(robot, foci_result.initial_trajectory)

    foci_metrics = planner_metrics(foci_result, foci_min_dist, foci_path_length, scene)
    foci_metrics["median_solve_time_s"] = float(foci_wall)
    save_results_csv(
        results_csv,
        scene="bonsai",
        robot=robot,
        trajectories={
            "FOCI": foci_result.trajectory,
            "RRT*": foci_result.initial_trajectory,
            "CHOMP": chomp_result.trajectory,
            "STOMP": stomp_result.trajectory,
        },
        metrics={
            "scene": {
                "scene_diagonal_m": scene,
                "n_environment_gaussians": len(obstacle_means),
                "n_robot_gaussians": len(gaussian_specs),
            },
            "FOCI": foci_metrics,
            "RRT*": {
                "total_time_s": float(foci_result.timings["initializer"]),
                "min_distance_m": float(rrt_min_dist),
                "path_length_m": float(rrt_path_length),
                "path_length_norm": float(rrt_path_length / scene),
            },
            "CHOMP": planner_metrics(chomp_result, chomp_min_dist, chomp_path_length, scene),
            "STOMP": planner_metrics(stomp_result, stomp_min_dist, stomp_path_length, scene),
        },
    )

    if RECORD_VIDEO:
        vis = RobotVisualizerOnline(
            robot=robot,
            trajectory=foci_result.trajectory,
            orbit_camera=True,
            orbit_period=ORBIT_PERIOD,
            **orbit_camera_params(robot, foci_result.trajectory, extra_points=[goal, np.zeros(3)]),
        )
        vis.visualize_goal(goal)
        vis.visualize_gaussian_splat("Bonsai", obstacle_means, obstacle_covs, colors, opacities)
        vis.visualize_robot_gaussians()
        vis.visualize_path(name="FOCI")

        # 1) FOCI trajectory only.
        vis.visualize_trajectory(
            loop=True,
            record=True,
            record_path=record_path_foci,
            record_fps=RECORD_FPS,
            record_duration=ORBIT_PERIOD,
        )

        # 2) Same scene and orbit with every planner's trajectory.
        vis.visualize_initializer_path(foci_result.initial_trajectory, name="RRT*", color=(255, 0, 0))
        vis.visualize_initializer_path(chomp_result.trajectory, name="CHOMP", color=(0, 0, 255))
        vis.visualize_initializer_path(stomp_result.trajectory, name="STOMP", color=(255, 0, 255))
        vis.visualize_trajectory(
            loop=True,
            record=True,
            record_path=record_path_all,
            record_fps=RECORD_FPS,
            record_duration=ORBIT_PERIOD,
        )
        return

    vis = RobotVisualizer(
        robot=robot,
        trajectory=foci_result.trajectory
    )

    vis.visualize_goal(goal)
    vis.visualize_gaussian_splat("Bonsai", obstacle_means, obstacle_covs, colors, opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_path(name="FOCI")
    vis.visualize_initializer_path(foci_result.initial_trajectory, name="RRT*", color=(255, 0, 0))
    vis.visualize_initializer_path(chomp_result.trajectory, name="CHOMP", color=(0, 0, 255))
    vis.visualize_initializer_path(stomp_result.trajectory, name="STOMP", color=(255, 0, 255))
    vis.visualize_trajectory(loop=True)

            
if __name__ == "__main__":
    benchmark_bonsai()