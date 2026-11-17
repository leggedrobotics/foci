import numpy as np

from src.robots.drone import DroneRobot, DroneGaussian
from src.robots.mobile import MobileRobot, MobileGaussian
from src.robots.manipulator import ManipulatorRobot, LinkGaussian
# from src.environment.environment import GaussianEnvironment
from src.environment.obstacle import StaticObstacle, MobileObstacle
from src.environment.environment import GaussianEnvironmentOnline
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner_online import OnlinePlanner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizerOnline
from src.benchmark.utils import minimum_robot_environment_distance


def mobile_robot_demo():

    num_samples = 25

    trajectory = np.array([(i * 0.1, 0.0, 0.0) for i in range(num_samples)])

    obstacles = [
        StaticObstacle(np.array([2.0, 0.0, 1.0]), np.diag([0.12**2, 0.12**2, 0.6**2])),
        StaticObstacle(np.array([0.0, 2.0, 1.0]), np.diag([0.12**2, 0.12**2, 0.6**2])),
        StaticObstacle(np.array([2.0, 2.5, 1.0]), np.diag([0.12**2, 0.12**2, 0.6**2])),
        MobileObstacle(np.array([1.0, 1.0, 1.0]), np.diag([0.12**2, 0.12**2, 0.6**2]), trajectory),
    ]

    environment = GaussianEnvironmentOnline(obstacles)

    gaussian_specs = [ 
            LinkGaussian(5, 0.5, np.diag([0.1, 0.02, 0.2])), 
            LinkGaussian(6, 0.5, np.diag([0.1, 0.02, 0.2])), 
            LinkGaussian(7, 0.5, np.diag([0.1, 0.02, 0.05])), 
            LinkGaussian(8, 0.5, np.diag([0.1, 0.02, 0.05])), 
        ]
    
    theta_start = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    goal = np.array([3.0, 3.0, 4.0])

    robot = ManipulatorRobot(
        urdf_path="urdfs/ur5_extended_move.urdf",
        root_link="world",
        tip_link="ee_link",
        gaussian_specs=gaussian_specs,
    )

    joint_groups = JointGroups(
            virtual_indices=[0,1,2], 
            virtual_wmax=4.5, 
            virtual_amax=5.0, 
            real_wmax=4.5, 
            real_amax=5.5,
        )
    
    
#     initializer = RRTStarInitializer(
#         voxel_size=0.1,
#         goal_threshold=0.005,
#         random_seed=10,
#         max_time=None,   
# )

    planner = OnlinePlanner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        # initializer=initializer,
        num_control_points=12,
        num_samples=num_samples,
        weights={
            "goal": 10.0,
            "obstacle": 0.01,
            "jerk": 0.001,
            "virtual_jerk": 0.001,
        },
        vmax=5.0,
        linear_solver="mumps"
    )

    result = planner.plan(
        start=theta_start,
        goal=goal,
    )

    # min_dist_foci, info = minimum_robot_environment_distance(robot,environment,result.trajectory)

    print("\n--- Planning timings FOCI---")
    print(f"Initializer: {result.timings['initializer']:.6f} s")
    print(f"Build:       {result.timings['build']:.6f} s")
    print(f"Solver:      {result.timings['solve']:.6f} s")
    print(f"Total:       {result.timings['total']:.6f} s")
    print(f"Success:     {result.success}")

    # print("\n--- Benchmark metrics ---")
    # print(f"Number of environment gaussians: {len(obstacles)}")
    # print(f"Number of robot gaussians: {len(gaussian_specs)}")
    # print(f"Minimum robot-environment distance FOCI: {min_dist_foci:.3f} m")

    vis = RobotVisualizerOnline(
        robot=robot,
        trajectory=result.trajectory,

    )

    vis.visualize_goal(goal)
    vis.visualize_obstacles_online(environment)
    vis.visualize_robot_gaussians()
    vis.visualize_path()
    vis.visualize_initializer_path(result.initial_trajectory, color=(0, 0, 255), name="RRT*")
    vis.visualize_trajectory(loop=True)

    

if __name__ == "__main__":
    mobile_robot_demo()