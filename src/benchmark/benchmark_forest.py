import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data_2
from src.robots.drone import DroneGaussian, DroneRobot
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.initialize.straight_line import StraightLineInitializer
from src.initialize.uniform_initializer import UniformControlPointsInitializer
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

# RECORD_VIDEO = True records a video of the FOCI trajectory only, with the
# camera orbiting the scene once (ORBIT_PERIOD seconds). It needs a browser
# connected to the viser URL; the recording stops by itself after one orbit.
# With False, the interactive viewer shows FOCI, CHOMP and STOMP.
# Every run also writes its statistics and final trajectories to docs/results.csv
# (see save_results_csv), replacing this scene's previous rows.
RECORD_VIDEO = False

# FOCI's warm start: "uniform" (the Planner's default,
# UniformControlPointsInitializer: every control point at the start, as in
# demos/forest.py), "straight_line" (StraightLineInitializer) or "rrt"
# (RRTStarInitializer). CHOMP and STOMP start from the same trajectory,
# except with "uniform", where they use their own default (a straight line
# in configuration space to FOCI's final configuration).
FOCI_INITIALIZER = "straight_line"
RECORD_FPS = 10
ORBIT_PERIOD = 26.0

# Static viewer camera just behind the goal, looking back towards the
# start. Offsets are in a frame built from the horizontal start->goal
# direction: CAMERA_BACK metres past the goal along it, CAMERA_SIDE metres
# to its left and CAMERA_UP metres up. Straight behind the goal (side = 0)
# the start is hidden by a moss-covered trunk and the camera sits inside
# foliage, so the camera is shifted a little to the side; the look-at point
# is CAMERA_LOOK_FRAC of the way from goal to start, nudged CAMERA_LOOK_SIDE
# metres sideways, so the goal, the start and the paths' detours all fit.
# Note: from this distance the start sphere itself is mostly occluded by
# that trunk at wide FOVs (the paths visibly converge on it).
CAMERA_BACK = 3.0
CAMERA_UP = 3.0
CAMERA_SIDE = 2.25
CAMERA_LOOK_FRAC = 0.45
CAMERA_LOOK_SIDE = -1.0
CAMERA_FOV_DEG = 60.0


def behind_goal_camera(start_xyz, goal_xyz):
    """Camera position and look-at point just behind the goal, facing the start."""
    start_xyz = np.asarray(start_xyz, dtype=float)
    goal_xyz = np.asarray(goal_xyz, dtype=float)
    forward = goal_xyz - start_xyz
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    position = goal_xyz + CAMERA_BACK * forward + CAMERA_SIDE * left + CAMERA_UP * up
    look_at = goal_xyz + CAMERA_LOOK_FRAC * (start_xyz - goal_xyz) + CAMERA_LOOK_SIDE * left
    return position, look_at


def benchmark_forest():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    record_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "benchmark_forest.mp4"))
    results_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "results.csv"))
    ply_file = os.path.join(PROJECT_ROOT, "data/Forest.ply")

    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)

    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    translation = np.array([8.0, 0.0, 0.0])
    scale_factor = 10.0
    
    
    obstacle_means = (obstacle_means * scale_factor) @ rotation.T + translation
    obstacle_covs = np.einsum("ij,njk,lk->nil", rotation, obstacle_covs, rotation) * scale_factor**2


    gaussian_specs = [DroneGaussian(np.array([0.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.1 ** 2)]

    # Same query as demos/forest.py: the straight line goes through the large
    # leaning trunk near (7, 0); both endpoints are > 1 m from any splat.
    # Previous query: start (0, -3, 1), goal (14, 5, 3.5).
    theta_start = np.array([-2.19, -0.11, 3.12, 0.0])
    goal = np.array([13.24, 0.91, 3.6])

    # yaw_limits widened from the default (-pi, pi): this robot's single
    # Gaussian is centred at the body with no offset, so yaw affects neither
    # collision nor goal-reaching at all. A tight +-pi box on the B-spline
    # control points still forbids any trajectory whose yaw would need to
    # cross that wrap boundary (even though the underlying angle is
    # periodic), which is what was forcing IPOPT into much harder, longer
    # detours -- confirmed by isolating yaw's bound from xyz's: yaw alone
    # took 173 IPOPT iterations and only reached "Acceptable", vs 24 and
    # "Optimal" for xyz's bounds alone, and combined they hit the 1000-iter
    # cap without converging at all. That was with the RRT* warm start; the
    # other warm starts keep the default (-pi, pi), as demos/forest.py does,
    # so FOCI reproduces the demo's trajectory exactly.
    robot = DroneRobot(
        urdf_path="urdfs/drone_example.urdf",
        arm_length=0.15,
        gaussian_specs=gaussian_specs,
        yaw_limits=(-10 * np.pi, 10 * np.pi) if FOCI_INITIALIZER == "rrt" else (-np.pi, np.pi),
    )

    joint_groups = JointGroups(
        virtual_indices=[0,1,2], 
        virtual_wmax=5.0, 
        virtual_amax=3.0, 
        real_wmax=6.5, 
        real_amax=3.5,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    # Back to RRT*: with yaw_limits widened and the NLP/solver now cached
    # across plan() calls (see Planner._cache_built_problem), the previous
    # failure mode (IPOPT hitting its 1000-iter cap on a yaw-constrained
    # warm start) should no longer apply, so it's worth seeing what RRT*'s
    # own (differently-shaped, obstacle-aware) warm start gives FOCI versus
    # the straight-line one CHOMP/STOMP still use. RRT* itself still costs
    # ~8-11s here (2.4M obstacles, non-sampleable task-space goal) and
    # rarely finds an exact solution -- that part is unaffected by either
    # fix above.
    if FOCI_INITIALIZER == "rrt":
        initializer = RRTStarInitializer(
            voxel_size=0.01,
            goal_threshold=0.005,
            random_seed=42,
            max_time=None,
        )
        init_label = "RRT*"
    elif FOCI_INITIALIZER == "straight_line":
        initializer = StraightLineInitializer()
        init_label = "Straight line"
    elif FOCI_INITIALIZER == "uniform":
        initializer = UniformControlPointsInitializer()
        init_label = "Uniform (start)"
    else:
        raise ValueError(f"Unknown FOCI_INITIALIZER {FOCI_INITIALIZER!r}")

    foci_planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
        num_control_points=12,
        num_samples=25,
        # Same FOCI parameters (and query, joint limits and straight-line warm
        # start) as demos/forest.py.
        weights={
            "goal": 0.75,
            "obstacle": 500.0,
            "jerk": 0.001,
            "virtual_jerk": 0.001,
        },
        vmax=1.0,
        linear_solver="ma27"
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

    chomp_planner = CHOMPPlanner(
        robot=robot, 
        environment=environment, 
        joint_groups=joint_groups, 
        num_waypoints=12,
        max_iter=1_000, 
        learning_rate=0.01, 
        weights={"obstacle": 1.0, "smoothness": 1.0},
        convergence_tol=1e-3,            
    )

    # FOCI only: 1 untimed warmup + median of 3 timed calls, matching the
    # FOCI paper's eval/baselines.py methodology (_median_time) so this
    # number stays comparable with the earlier experiments. CHOMP/STOMP are
    # called once: both build their CasADi/warp machinery once in __init__
    # (outside this call), and since FOCI already ran first in this same
    # process, warp's kernel cache (shared across ConvolutionFunctor
    # instances, process-wide) is already warm by the time these run -- a
    # warmup here would just re-pay their solve for no JIT-avoidance
    # benefit, only extra wall time.
    foci_result, foci_wall = median_solve_time(
        lambda: foci_planner.plan(start=theta_start, goal=goal)
    )

    theta_final = foci_result.trajectory[-1]

    # A constant "uniform" warm start would make CHOMP/STOMP jump from the
    # start to the goal in their last waypoint; give them their own default
    # (straight line in configuration space) instead.
    baseline_init = None if FOCI_INITIALIZER == "uniform" else foci_result.initial_trajectory

    stomp_result = stomp_planner.plan(
        start=theta_start,
        goal=theta_final,
        initial_trajectory=baseline_init,
    )

    chomp_result = chomp_planner.plan(
        start=theta_start,
        goal=theta_final,
        initial_trajectory=baseline_init,
    )

    # opacities=opacities drops near-transparent splats before measuring
    # clearance (opacity <= 0.5), matching the paper eval's OPACITY_MIN.
    foci_min_dist, info = minimum_robot_environment_distance(
        robot, environment, foci_result.trajectory, opacities=opacities, include_robot_covariance=False
    )
    stomp_min_dist, info = minimum_robot_environment_distance(
        robot, environment, stomp_result.trajectory, opacities=opacities, include_robot_covariance=False
    )
    chomp_min_dist, info = minimum_robot_environment_distance(
        robot, environment, chomp_result.trajectory, opacities=opacities, include_robot_covariance=False
    )
    # FOCI's own warm start (the initializer's path, before IPOPT).
    rrt_min_dist, info = minimum_robot_environment_distance(
        robot, environment, foci_result.initial_trajectory, opacities=opacities, include_robot_covariance=False
    )

    foci_path_length = task_path_length(robot, foci_result.trajectory)
    stomp_path_length = task_path_length(robot, stomp_result.trajectory)
    chomp_path_length = task_path_length(robot, chomp_result.trajectory)
    rrt_path_length = task_path_length(robot, foci_result.initial_trajectory)

    # Dimensionless, comparable across scenes of different physical size
    # (the paper eval reports length / scene_size, not a raw metre figure).
    scene = scene_size(obstacle_means, trim_percent=2.0)

    print("\n--- Planning timings FOCI---")
    print(f"Median solve (3 reps, 1 warmup): {foci_wall:.6f} s")
    print(f"Last call -- Initial guess: {foci_result.timings['initializer']:.6f} s")
    print(f"Last call -- Build:  {foci_result.timings['build']:.6f} s")
    print(f"Last call -- Solver: {foci_result.timings['solve']:.6f} s")
    print(f"Last call -- Total:  {foci_result.timings['total']:.6f} s")
    print(f"Success:     {foci_result.success}")

    print("\n--- Planning timings STOMP---")
    print(f"Initial guess: {stomp_result.timings['initializer']:.6f} s")
    print(f"Build:  {stomp_result.timings['build']:.6f} s")
    print(f"Solver: {stomp_result.timings['solve']:.6f} s")
    print(f"Total:  {stomp_result.timings['total']:.6f} s")
    print(f"Success:     {stomp_result.success}")

    print("\n--- Planning timings CHOMP---")
    print(f"Initial guess: {chomp_result.timings['initializer']:.6f} s")
    print(f"Build:  {chomp_result.timings['build']:.6f} s")
    print(f"Solver: {chomp_result.timings['solve']:.6f} s")
    print(f"Total:  {chomp_result.timings['total']:.6f} s")
    print(f"Success:     {chomp_result.success}")

    print(f"\n--- Planning timings {init_label} (FOCI initializer, last call)---")
    print(f"Total:  {foci_result.timings['initializer']:.6f} s")

    print("\n--- Benchmark metrics ---")
    print(f"Number of environment gaussians: {len(obstacle_means)}")
    print(f"Number of robot gaussians: {len(gaussian_specs)}")
    print(f"Minimum robot-environment distance FOCI: {foci_min_dist:.3f} m")
    print(f"Minimum robot-environment distance STOMP: {stomp_min_dist:.3f} m")
    print(f"Minimum robot-environment distance CHOMP: {chomp_min_dist:.3f} m")
    print(f"Minimum robot-environment distance {init_label}: {rrt_min_dist:.3f} m")
    print(f"Path length FOCI:  {foci_path_length:.3f} m  ({foci_path_length / scene:.3f} x scene diagonal)")
    print(f"Path length STOMP: {stomp_path_length:.3f} m  ({stomp_path_length / scene:.3f} x scene diagonal)")
    print(f"Path length CHOMP: {chomp_path_length:.3f} m  ({chomp_path_length / scene:.3f} x scene diagonal)")
    print(f"Path length {init_label}:  {rrt_path_length:.3f} m  ({rrt_path_length / scene:.3f} x scene diagonal)")


    foci_metrics = planner_metrics(foci_result, foci_min_dist, foci_path_length, scene)
    foci_metrics["median_solve_time_s"] = float(foci_wall)
    save_results_csv(
        results_csv,
        scene="forest",
        robot=robot,
        trajectories={
            "FOCI": foci_result.trajectory,
            init_label: foci_result.initial_trajectory,
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
            init_label: {
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
            **orbit_camera_params(robot, foci_result.trajectory, extra_points=[goal, theta_start[:3]]),
        )
        vis.visualize_goal(goal)
        vis.visualize_gaussian_splat("Forest", obstacle_means, obstacle_covs, colors, opacities)
        vis.visualize_robot_gaussians()
        vis.visualize_path(name="FOCI")
        vis.visualize_trajectory(
            loop=True,
            record=True,
            record_path=record_path,
            record_fps=RECORD_FPS,
            record_duration=ORBIT_PERIOD,
        )
        return

    vis = RobotVisualizer(
        robot=robot,
        trajectory=foci_result.trajectory
    )

    vis.visualize_goal(goal, radius=0.15)
    vis.visualize_gaussian_splat("Forest", obstacle_means, obstacle_covs, colors, opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_path(name="FOCI")
    vis.visualize_initializer_path(stomp_result.trajectory, name="STOMP", color=(255, 0, 255))
    vis.visualize_initializer_path(chomp_result.trajectory, name="CHOMP", color=(0, 0, 255))
    vis.visualize_initializer_path(foci_result.initial_trajectory, name=init_label, color=(255, 0, 0))
    vis.server.scene.add_icosphere(
        name="start", radius=0.3, color=(255, 255, 0), position=theta_start[:3].astype(float),
    )

    camera_position, camera_look_at = behind_goal_camera(theta_start[:3], goal)

    @vis.server.on_client_connect
    def _(client):
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = camera_position
        client.camera.look_at = camera_look_at
        client.camera.fov = np.deg2rad(CAMERA_FOV_DEG)

    vis.visualize_trajectory(loop=True)

    
if __name__ == "__main__":
    benchmark_forest()