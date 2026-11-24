import os

import numpy as np

from src.utils.ply import extract_splat_data_2
from src.robots.mobile import MobileRobot, MobileGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance

RECORD_VIDEO = False
RECORD_FPS = 10

# Scale factor applied to the raw splat (reconstruction is sub-meter).
SCALE_FACTOR = 20.0
BODY_HEIGHT = 0.5

# Number of points placed around the stone circle and number of
# start -> goal crossings generated from them (point i -> point i + NUM_POINTS // 2).
NUM_POINTS = 12
NUM_PATHS = NUM_POINTS // 2

PATH_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
]


def stonehenge_demo():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    record_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "stonehenge.mp4"))
    ply_file = os.path.join(PROJECT_ROOT, "data", "stonehenge.ply")

    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)

    obstacle_means = obstacle_means * SCALE_FACTOR
    obstacle_covs = obstacle_covs * SCALE_FACTOR**2

    radius = max(np.linalg.norm(obstacle_means[:, :2], axis=1)) * 1.02

    gaussian_specs = [MobileGaussian(np.array([0.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.1**2)]

    robot = MobileRobot(
        urdf_path=None,
        body_center_height=BODY_HEIGHT,
        gaussian_specs=gaussian_specs,
    )

    joint_groups = JointGroups(
        virtual_indices=[0, 1],
        virtual_wmax=6.0,
        virtual_amax=3.5,
        real_wmax=2.0,
        real_amax=1.5,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    initializer = RRTStarInitializer(
        voxel_size=0.1,
        goal_threshold=0.1,
        random_seed=10,
        max_time=None,
    )

    planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
        num_control_points=12,
        num_samples=25,
        weights={
            "goal": 5.0,
            "obstacle": 2.0,
            "jerk": 0.001,
            "virtual_jerk": 0.001,
        },
        vmax=5.0,
        linear_solver="ma27",
    )

    # 12 points around the stone circle; path i crosses from point i to
    # point i + NUM_PATHS, through the middle of the circle.
    points = []
    for i in range(NUM_POINTS):
        theta = i * 2 * np.pi / NUM_POINTS
        points.append([radius * np.cos(theta), radius * np.sin(theta)])

    results = []
    for i in range(NUM_PATHS):
        start_x, start_y = points[i]
        goal_x, goal_y = points[i + NUM_PATHS]

        theta_start = np.array([start_x, start_y, np.pi / 2])
        goal = np.array([goal_x, goal_y, BODY_HEIGHT])

        result = planner.plan(start=theta_start, goal=goal)

        print(f"\n--- Path {i} ({i} -> {i + NUM_PATHS}) ---")
        print(f"Initializer: {result.timings['initializer']:.6f} s")
        print(f"Build:       {result.timings['build']:.6f} s")
        print(f"Solver:      {result.timings['solve']:.6f} s")
        print(f"Total:       {result.timings['total']:.6f} s")
        print(f"Success:     {result.success}")

        results.append(result)

    successful = [r for r in results if r.success]
    print(f"\n--- Stonehenge: {len(successful)}/{NUM_PATHS} paths succeeded ---")

    if not successful:
        print("No successful trajectory to visualize, aborting.")
        return

    anim_trajectory = successful[0].trajectory

    min_dist, info = minimum_robot_environment_distance(robot, environment, anim_trajectory)
    print(f"Minimum robot-environment distance (path 0): {min_dist:.3f} m")

    vis = RobotVisualizer(
        robot=robot,
        trajectory=anim_trajectory,
    )

    vis.visualize_gaussian_splat("Stonehenge", obstacle_means, obstacle_covs, colors, opacities)
    vis.visualize_robot_gaussians()

    for i, result in enumerate(results):
        if not result.success:
            continue
        color = PATH_COLORS[i % len(PATH_COLORS)]
        vis.visualize_path(color=color, name=f"Path {i}")

    vis.visualize_trajectory(loop=True, record=RECORD_VIDEO, record_path=record_path, record_fps=RECORD_FPS)


if __name__ == "__main__":
    stonehenge_demo()
