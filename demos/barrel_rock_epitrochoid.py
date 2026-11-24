import sys
import select
import time
import os

import numpy as np
import imageio.v3 as iio

from src.utils.ply import extract_splat_data
from src.robots.drone import StandardDroneRobot, DroneGaussian
from src.environment.environment import GaussianEnvironmentOnline
from src.environment.obstacle import StaticObstacle, MobileObstacle
from src.planning.planner_online import OnlinePlanner
from src.planning.joints import JointGroups
from src.initialize.straight_line import StraightLineInitializer
from src.visualization.visualizer import RobotVisualizerOnline
from src.benchmark.utils import save_results_csv, scene_size, task_path_length


# Both source clouds are dense captured point clouds (one Gaussian per point);
# cap each at this many Gaussians so viser can render the scene smoothly.
TARGET_MAX_POINTS = 80_000

N_BARRELS = 7
BARREL_SIZE_FRACTION = 0.10   # barrel size relative to the rock's size
ROCK_SCALE = 2.2              # enlarge the rock beyond its raw captured size
ORBIT_CLEARANCE = 0.25        # extra radius added beyond the rock's footprint
ORBIT_SWEEP = np.radians(130) # how far each barrel travels along its epitrochoid during the flight

# Four barrels sit at the four corners of a rhombus around the rock --
# front/back on the start -> goal line (so the drone meets one right after
# takeoff and the other right before reaching the goal) and left/right
# pulled closer to each other, completing the rhombus. Three more barrels
# fly well above the rest: one right over the rhombus's center, one over
# its back-left corner, one over its front-right corner. Each barrel still
# moves on its own small epitrochoid loop around its spot (its own lobe
# count, loop sharpness and sweep direction) instead of sitting still.
BARREL_LINE_MARGIN = 1.95      # distance from the rock's edge to each barrel's loop center
BARREL_SIDE_FACTOR = 0.55      # left/right barrels pulled closer together than front/back -> rhombus, not a square
HIGH_BARREL_SPREAD = 0.4       # how far the back-left/front-right "above" barrels sit from the center one -> the three stay close together
BARREL_LOOP_RADIUS = 0.55      # size of each barrel's local epitrochoid loop
HIGH_BARREL_HEIGHT_FRACTION = 2.15 # shared height of the three "above" barrels (fraction of the rock's height)
BARREL_HEIGHT_FRACTIONS = np.array([
    0.9, 0.65, 0.775, 0.775,                                   # front, back, left, right
    HIGH_BARREL_HEIGHT_FRACTION, HIGH_BARREL_HEIGHT_FRACTION, HIGH_BARREL_HEIGHT_FRACTION,  # center, back-left, front-right -- all above
])

# Randomized (fixed seed, so the scene is still reproducible) per-barrel loop
# shape instead of a smooth progression -- lobe count, loop sharpness,
# starting phase and sweep direction all vary unevenly, so the barrels don't
# look like they're all tracing the same loop a beat apart.
_barrel_rng = np.random.default_rng(42)
BARREL_LOBES = _barrel_rng.integers(3, 9, size=N_BARRELS)             # R/r ratio -> number of lobes
BARREL_LOOPINESS = _barrel_rng.uniform(0.3, 0.8, size=N_BARRELS)      # d/r ratio -> loop sharpness
BARREL_PHASE = _barrel_rng.uniform(0.0, 2 * np.pi, size=N_BARRELS)    # starting angle along each barrel's loop
BARREL_SWEEP_SIGNS = _barrel_rng.choice([-1.0, 1.0], size=N_BARRELS)

NUM_SAMPLES = 24

# extract_splat_data has no real per-point scale (this .ply has no scale_*
# columns), so it fills covariances with sigma = 1 m identity -- on a multi-
# metre object that is one giant blob per point, totally overlapping. Shrink
# each point to a plausible dot radius before display. It is kept fixed
# (not tied to the barrel's own size) so the dots don't shrink below a
# visible size once the barrel itself is scaled down.
POINT_SIGMA = 0.01

# A point cloud's variance underestimates the true half-extent of a filled
# solid (e.g. ~1/3 of it, for a uniform-filled box), so inflate the fitted
# covariance when approximating a splat as a single Gaussian obstacle for
# planning -- otherwise the optimizer would see a "safe" region that is
# actually still inside the object.
OBSTACLE_INFLATION = 3.5

# Set to True to record the visualization to an mp4 instead of (or besides)
# just watching it live -- frames are grabbed from a connected browser client,
# so a client must be connected before recording can start.
RECORD_VIDEO = False
RECORD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "barrel_rock_epitrochoid.mp4"))
RECORD_FPS = 15

# Statistics and final trajectory are written to the shared benchmark
# results file (rows of scene "barrel_rock_epitrochoid" are replaced).
RESULTS_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "results.csv"))

# Camera slowly orbits the whole rock + barrels course once over
# ORBIT_PERIOD seconds (independent of the drone/barrel animation, which
# keeps looping underneath at its own faster pace) so the recording shows
# the full scene rather than just chasing the drone. Recording stops
# automatically once the orbit completes.
ORBIT_PERIOD = 26.0

# Fraction of the default orbit distance (radius and height scaled together,
# so the viewing angle stays the same): < 1 brings the camera closer.
ORBIT_ZOOM = 0.75


def _splat_size(means: np.ndarray) -> float:
    """Bounding-box diagonal of a point cloud -- the "size" reported for a 3DGS."""
    extent = means.max(axis=0) - means.min(axis=0)
    return float(np.linalg.norm(extent))


def _random_subsample(
    means: np.ndarray,
    covariances: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    keep_fraction: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep a random keep_fraction of the Gaussians, without replacement."""
    n = means.shape[0]
    n_keep = max(1, int(round(n * keep_fraction)))
    idx = np.random.default_rng(seed).choice(n, size=n_keep, replace=False)
    return means[idx], covariances[idx], colors[idx], opacities[idx]


def _ground_and_center(means: np.ndarray) -> np.ndarray:
    """Translate a point cloud so it is centered on x/y and resting on z = 0."""
    centroid_xy = means[:, :2].mean(axis=0)
    min_z = means[:, 2].min()
    return means - np.array([centroid_xy[0], centroid_xy[1], min_z])


def _epitrochoid_point(theta: float, R: float, r: float, d: float) -> tuple[float, float]:
    """
    Epitrochoid: traced by a point at distance d from the center of a circle
    of radius r rolling, without slipping, on the outside of a fixed circle
    of radius R.
    """
    k = (R + r) / r
    x = (R + r) * np.cos(theta) - d * np.cos(k * theta)
    y = (R + r) * np.sin(theta) - d * np.sin(k * theta)
    return x, y


def _fit_obstacle_gaussian(means: np.ndarray, inflation: float = OBSTACLE_INFLATION) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a point cloud into a single Gaussian (mean + inflated diagonal
    covariance) usable as a planning obstacle -- the dense splat itself has
    far too many points to hand to the online planner one by one."""
    mean = means.mean(axis=0)
    cov = np.diag(means.var(axis=0)) * inflation
    return mean, cov


def barrel_rock_epitrochoid_demo():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rock_file = os.path.join(PROJECT_ROOT, "data", "Rock on forest path.ply")
    barrel_file = os.path.join(PROJECT_ROOT, "data", "Barrel.ply")

    rock_means, rock_covs, rock_colors, rock_opacities = extract_splat_data(rock_file)
    barrel_means, barrel_covs, barrel_colors, barrel_opacities = extract_splat_data(barrel_file)

    rock_size = _splat_size(rock_means)
    barrel_size = _splat_size(barrel_means)

    print("--- 3DGS analysis (raw, before filtering/resizing) ---")
    print(f"Rock:   {len(rock_means):>8} gaussians, bounding-box size (diagonal) = {rock_size:.3f} m")
    print(f"Barrel: {len(barrel_means):>8} gaussians, bounding-box size (diagonal) = {barrel_size:.3f} m")

    # if len(rock_means) > TARGET_MAX_POINTS:
    #     keep_fraction = TARGET_MAX_POINTS / len(rock_means)
    #     rock_means, rock_covs, rock_colors, rock_opacities = _random_subsample(
    #         rock_means, rock_covs, rock_colors, rock_opacities, keep_fraction
    #     )
    #     print(f"\nRock is too dense -> random filter kept {keep_fraction:.1%} "
    #           f"({len(rock_means)} gaussians).")

    # if len(barrel_means) > TARGET_MAX_POINTS:
    #     keep_fraction = TARGET_MAX_POINTS / len(barrel_means)
    #     barrel_means, barrel_covs, barrel_colors, barrel_opacities = _random_subsample(
    #         barrel_means, barrel_covs, barrel_colors, barrel_opacities, keep_fraction
    #     )
    #     print(f"Barrel is too dense -> random filter kept {keep_fraction:.1%} "
    #           f"({len(barrel_means)} gaussians).")

    rock_size *= ROCK_SCALE
    print(f"\nEnlarging rock by {ROCK_SCALE:.3f}x -> new size = {rock_size:.3f} m.")

    barrel_scale = (BARREL_SIZE_FRACTION * rock_size) / barrel_size
    print(f"Scaling barrel by {barrel_scale:.3f}x -> new size = "
          f"{barrel_scale * barrel_size:.3f} m ({BARREL_SIZE_FRACTION:.0%} of the rock).")

    rock_means = _ground_and_center(rock_means) * ROCK_SCALE
    rock_covs = np.tile(np.eye(3) * POINT_SIGMA**2, (len(rock_means), 1, 1))

    barrel_means = _ground_and_center(barrel_means) * barrel_scale
    barrel_covs = np.tile(np.eye(3) * POINT_SIGMA**2, (len(barrel_means), 1, 1))

    rock_half_extent_xy = float(
        (rock_means[:, :2].max(axis=0) - rock_means[:, :2].min(axis=0)).max() / 2.0
    )
    rock_height = float(rock_means[:, 2].max())

    # --- Simplified Gaussian obstacles for planning (the dense splats above
    # are for display only -- far too many points to hand to the online
    # planner directly). ---

    rock_obstacle_mean, rock_obstacle_cov = _fit_obstacle_gaussian(rock_means)
    rock_obstacle = StaticObstacle(rock_obstacle_mean, rock_obstacle_cov)

    barrel_obstacle_mean_local, barrel_obstacle_cov = _fit_obstacle_gaussian(barrel_means)

    line_offset = rock_half_extent_xy + ORBIT_CLEARANCE + BARREL_LINE_MARGIN
    side_offset = line_offset * BARREL_SIDE_FACTOR
    # Front/back stay on the start -> goal line; left/right are pulled closer
    # to each other, so the four barrels form a rhombus around the rock
    # instead of a square. The three "above" barrels sit near the rhombus's
    # center -- the back-left/front-right ones pulled in by HIGH_BARREL_SPREAD
    # so all three stay close together -- their height (in
    # BARREL_HEIGHT_FRACTIONS) is what places them above the rest.
    barrel_square_centers = np.array([
        [-line_offset, 0.0],                                                     # front
        [line_offset, 0.0],                                                      # back
        [0.0, -side_offset],                                                     # left
        [0.0, side_offset],                                                      # right
        [0.0, 0.0],                                                              # center, above
        [line_offset * HIGH_BARREL_SPREAD, -side_offset * HIGH_BARREL_SPREAD],   # back-left, above
        [-line_offset * HIGH_BARREL_SPREAD, side_offset * HIGH_BARREL_SPREAD],   # front-right, above
    ])

    barrel_base_xyz = []
    barrel_trajectories = []
    barrel_obstacles = []

    for i in range(N_BARRELS):
        orbit_r = BARREL_LOOP_RADIUS / float(BARREL_LOBES[i])
        orbit_d = orbit_r * float(BARREL_LOOPINESS[i])
        height = float(BARREL_HEIGHT_FRACTIONS[i]) * rock_height

        sweep = float(BARREL_SWEEP_SIGNS[i]) * ORBIT_SWEEP
        thetas = float(BARREL_PHASE[i]) + np.linspace(0.0, sweep, NUM_SAMPLES)
        xy = np.array([_epitrochoid_point(theta, BARREL_LOOP_RADIUS, orbit_r, orbit_d) for theta in thetas])
        xy += barrel_square_centers[i]

        base_xyz = np.array([xy[0, 0], xy[0, 1], height])
        trajectory = np.column_stack([xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1], np.zeros(NUM_SAMPLES)])

        barrel_base_xyz.append(base_xyz)
        barrel_trajectories.append(trajectory)
        barrel_obstacles.append(
            MobileObstacle(barrel_obstacle_mean_local + base_xyz, barrel_obstacle_cov, trajectory)
        )

    environment = GaussianEnvironmentOnline(obstacles=[rock_obstacle] + barrel_obstacles)

    # Scene size for the normalised path length, as for the other benchmarks
    # (XY diagonal of the scene's bounding box): the box holding the rock and
    # every barrel at every position along its loop, i.e. all the space the
    # dynamic scene occupies during the trajectory.
    scene_corners = [rock_means.min(axis=0), rock_means.max(axis=0)]
    barrel_low, barrel_high = barrel_means.min(axis=0), barrel_means.max(axis=0)
    for base_xyz, trajectory in zip(barrel_base_xyz, barrel_trajectories):
        positions = base_xyz + trajectory
        scene_corners += [barrel_low + positions.min(axis=0), barrel_high + positions.max(axis=0)]
    scene = scene_size(np.array(scene_corners))

    gaussian_specs = [DroneGaussian(np.array([0.0, 0.0, 0.0]), np.diag([0.25**2, 0.35**2, 0.12**2]))]
    robot = StandardDroneRobot(urdf_path="urdfs/drone_example.urdf", gaussian_specs=gaussian_specs)

    joint_groups = JointGroups(
        virtual_indices=[0, 1, 2],
        virtual_wmax=8.0,
        virtual_amax=6.0,
        real_wmax=4.0,
        real_amax=6.0,
    )

    planner = OnlinePlanner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=StraightLineInitializer(),
        num_control_points=12,
        num_samples=NUM_SAMPLES,
        weights={
            "goal": 8.0,
            "obstacle": 4.5,
            "jerk": 0.03,
            "virtual_jerk": 0.05,
        },
        vmax=2.0,
        linear_solver="mumps",
    )

    # Start and goal sit on opposite sides of the rock, beyond both barrel
    # loops, at the obstacle course's median height -- cruising straight
    # across at that altitude runs into the rock and both barrels, so the
    # drone actually has to weave between them.
    flight_radius = line_offset + BARREL_LOOP_RADIUS + 1.4
    flight_height = 0.7 * rock_height
    # Start noticeably above the median course height so the drone can't just
    # duck under the rock and barrels at a low, trivial altitude -- it has to
    # come down into the obstacle course to reach the goal.
    start_height = 1.1 * rock_height
    start = np.array([-flight_radius, 0.0, start_height, 0.0])
    goal = np.array([flight_radius, 0.0, flight_height])

    result = planner.plan(start=start, goal=goal)
    duration = max(float(np.linalg.norm(goal - start[:3])) / planner.vmax, 1e-3)

    print(f"\n--- StandardDroneRobot through the rock + {N_BARRELS} barrels on the flight line ---")
    print(f"Success: {result.success}  ({result.solver_stats.get('return_status')})")
    print(f"Solve time: {result.timings['solve']:.3f} s")
    length = task_path_length(robot, result.trajectory)
    print(f"Path length: {length:.3f} m  ({length / scene:.3f} x scene diagonal, {scene:.3f} m)")

    save_results_csv(
        os.path.abspath(RESULTS_CSV),
        scene="barrel_rock_epitrochoid",
        robot=robot,
        trajectories={"FOCI": result.trajectory},
        metrics={
            "scene": {
                "scene_diagonal_m": scene,
                "n_environment_gaussians": len(environment.obstacles),
                "n_robot_gaussians": len(gaussian_specs),
            },
            "FOCI": {
                "success": bool(result.success),
                "return_status": result.solver_stats.get("return_status"),
                **{f"{key}_time_s": float(value) for key, value in result.timings.items()},
                "path_length_m": float(length),
                "path_length_norm": float(length / scene),
                "duration_s": float(duration),
            },
        },
    )

    # Orbit camera: centered on the course at its median height, wide enough
    # to keep the rhombus of barrels in frame, high enough to look slightly
    # down on the three barrels flying above the rest.
    orbit_center = np.array([0.0, 0.0, flight_height])
    orbit_radius = ORBIT_ZOOM * (flight_radius + 1.0)
    orbit_height = ORBIT_ZOOM * HIGH_BARREL_HEIGHT_FRACTION * rock_height * 1.15

    vis = RobotVisualizerOnline(
        robot=robot,
        trajectory=result.trajectory,
        duration=duration,
        orbit_camera=True,
        orbit_center=orbit_center,
        orbit_radius=orbit_radius,
        orbit_height=orbit_height,
        orbit_period=ORBIT_PERIOD,
    )
    vis.visualize_goal(goal)
    vis.visualize_robot_gaussians()
    vis.visualize_path()

    vis.server.scene.add_gaussian_splats(
        name="Rock",
        centers=rock_means,
        covariances=rock_covs,
        rgbs=rock_colors,
        opacities=rock_opacities,
    )

    barrel_handles = [
        vis.server.scene.add_gaussian_splats(
            name=f"Barrel {i}",
            centers=barrel_means,
            covariances=barrel_covs,
            rgbs=barrel_colors,
            opacities=barrel_opacities,
            position=barrel_base_xyz[i],
        )
        for i in range(N_BARRELS)
    ]

    print("\nVisualizing scene. Press 'q' + Enter to quit.")

    frames = []
    client = None
    if RECORD_VIDEO:
        print(f"Waiting for a browser client to connect before recording to {RECORD_PATH} ...")
        while len(vis.server.get_clients()) == 0:
            time.sleep(0.1)
        client = next(iter(vis.server.get_clients().values()))
        print(f"Client connected -- recording until you quit (fps = {RECORD_FPS}).")

    sample_idx = 0
    step = 0

    try:
        while True:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline().strip().lower() == "q":
                    print("Stopping visualization.")
                    break

            elapsed_time = step * 0.1

            q = result.trajectory[sample_idx]
            vis._update_robot(q, sample_idx)
            vis._update_robot_gaussians(sample_idx)
            vis._update_camera(sample_idx, elapsed_time)

            for i, handle in enumerate(barrel_handles):
                handle.position = barrel_base_xyz[i] + barrel_trajectories[i][sample_idx]

            if client is not None:
                frames.append(client.get_render(height=720, width=1280))
                if len(frames) % 20 == 0:
                    print(f"  ...captured {len(frames)} frames ({elapsed_time:.1f}/{ORBIT_PERIOD:.0f} s)")

            if RECORD_VIDEO and elapsed_time >= ORBIT_PERIOD:
                print(f"\nReached ORBIT_PERIOD = {ORBIT_PERIOD} s, stopping.")
                break

            time.sleep(0.1)

            step += 1
            sample_idx = (sample_idx + 1) % NUM_SAMPLES
    finally:
        if client is not None and frames:
            iio.imwrite(RECORD_PATH, frames, fps=RECORD_FPS)
            print(f"\nSaved {len(frames)} frames to {RECORD_PATH}.")


if __name__ == "__main__":
    barrel_rock_epitrochoid_demo()
