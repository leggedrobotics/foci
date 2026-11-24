"""
Generic visualization utilities for robot trajectories and Gaussian models.

This visualizer works with any robot implementing:

    robot.n_dof
    robot.f_task(q)
    robot.collision_points(q)
    robot.collision_covariances()

For manipulators, if the robot has a URDF path, the URDF is shown.
For drones, the trajectory and Gaussian collision model are shown directly.
"""

import sys
import select
import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from scipy.spatial.transform import Rotation
from viser import ViserServer
from viser.extras import ViserUrdf
from yourdfpy import URDF

from src.visualization.utils import EllipsoidFactory
from src.robots.drone import DroneRobot, finite_difference_acceleration


def _rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion in viser's (w, x, y, z) convention."""
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return np.array([w, x, y, z])


class RobotVisualizer:
    def __init__(
        self,
        robot,
        trajectory: np.ndarray,
        follow_camera: bool = False,
        camera_offset: np.ndarray | None = None,
        camera_lookahead: float = 0.5,
        duration: float | None = None,
    ):
        trajectory = np.asarray(trajectory, dtype=float)

        if trajectory.ndim != 2:
            raise ValueError(
                f"trajectory must have shape (num_samples, n_dof), "
                f"got {trajectory.shape}."
            )

        if trajectory.shape[1] != robot.n_dof:
            raise ValueError(
                f"trajectory has {trajectory.shape[1]} columns, "
                f"but robot.n_dof is {robot.n_dof}."
            )

        self.robot = robot

        self.trajectory = trajectory

        # Physical duration of the trajectory, needed to scale qdd (used by
        # robots with requires_acceleration = True) into real units -- the
        # unit-index-spacing fallback (duration=None) understates or
        # overstates the tilt by whatever factor the real dt differs from 1.
        self.duration = duration
        self._qdd = (
            finite_difference_acceleration(self.trajectory, dt=self._dt())
            if getattr(robot, "requires_acceleration", False)
            else None
        )

        self.follow_camera = follow_camera
        self.camera_offset = (
            np.asarray(camera_offset, dtype=float)
            if camera_offset is not None
            else np.array([0.0, 0.0, 1.5])
        )
        self.camera_lookahead = float(camera_lookahead)


        self.server = ViserServer()
        self.robot_urdf = robot.urdf_path

        self._ellipsoid_faces = self._create_ellipsoid_faces()
        self._robot_gaussian_handles = []
        self._ellipsoid_factory = None

        self.gaussian_points = self._compute_robot_gaussian_points()
        self.gaussian_covariances = np.asarray(
            robot.collision_covariances(),
            dtype=float,
        )

        if self.gaussian_covariances.shape[0] != self.gaussian_points.shape[1]:
            raise ValueError(
                "Number of collision covariances must match number of "
                "collision points. "
                f"Got {self.gaussian_covariances.shape[0]} covariances and "
                f"{self.gaussian_points.shape[1]} points."
            )

        if self.robot_urdf is not None: 
            if not Path(self.robot_urdf).is_file():
                raise FileNotFoundError(f"URDF file not found: {self.robot_urdf}")

            self.urdf = URDF.load(self.robot_urdf)
            self.viser_urdf = ViserUrdf(self.server, urdf_or_path=self.urdf, root_node_name="/robot")

            self.robot_frame = self.server.scene.add_frame("/robot",show_axes=False)

        # if self.follow_camera:
        #     q0 = self.trajectory[0]
        #     target0 = self.robot.f_task(q0)

        #     if hasattr(target0, "full"):
        #         target0 = target0.full()

        #     target0 = np.asarray(target0, dtype=float).reshape(3)

        #     self.server.initial_camera.position = target0 + self.camera_offset
        #     self.server.initial_camera.look_at = target0
        

    def _dt(self) -> float:
        """
        Seconds between consecutive trajectory samples, for scaling qdd into
        real units. Falls back to unit index spacing (dt=1) when this
        visualizer wasn't given the trajectory's physical duration -- that
        fallback is almost always wrong by a large factor for robots with
        requires_acceleration = True, so pass duration= when visualizing one.
        """
        if self.duration is None:
            return 1.0
        return self.duration / max(self.trajectory.shape[0] - 1, 1)

    def _compute_robot_gaussian_points(self) -> np.ndarray:
        """
        Evaluate robot.collision_points(q) for all trajectory samples.

        For robots with requires_acceleration = True (e.g. a standard
        drone), qdd is a finite-difference estimate over this trajectory --
        a best-effort tilt for display, not a physically exact one.

        Returns
        -------
        np.ndarray
            Shape (num_samples, n_gaussians, 3).
        """
        all_points = []

        for t, q in enumerate(self.trajectory):
            qdd = self._qdd[t] if self._qdd is not None else None
            points = self.robot.collision_points(q, qdd)

            if hasattr(points, "full"):
                points = points.full()

            points = np.asarray(points, dtype=float).reshape(-1, 3)
            all_points.append(points)

        return np.stack(all_points, axis=0)


    def visualize_goal(
        self,
        goal: np.ndarray,
        radius: float = 0.05,
        color: tuple[int, int, int] = (0, 0, 255),
    ):
        self.server.scene.add_icosphere(
            name="Goal",
            position=np.asarray(goal, dtype=float),
            radius=radius,
            color=color,
        )

    def visualize_obstacles(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        n_std: float = 2.0,
        color: tuple[int, int, int] = (255, 100, 100),
        opacity: float = 0.6,
        name: str = "Obstacle",
    ):
        means = np.asarray(means, dtype=float)
        covariances = np.asarray(covariances, dtype=float)

        factory = EllipsoidFactory(n_std=float(n_std))

        for obstacle_idx, (mean, cov) in enumerate(zip(means, covariances)):
            radii, quat_wxyz = factory.cov_to_ellipsoid(cov)

            self.server.scene.add_mesh_simple(
                name=f"{name}_{obstacle_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

    def visualize_robot_gaussians(
        self,
        n_std: float = 2.0,
        color: tuple[int, int, int] = (80, 160, 255),
        opacity: float = 0.35,
        name: str = "RobotGaussian",
    ):
        """
        Render robot collision Gaussians at the first trajectory sample.
        They are updated during animation.
        """
        self._robot_gaussian_handles = []
        self._ellipsoid_factory = EllipsoidFactory(n_std=float(n_std))

        for gaussian_idx in range(self.gaussian_points.shape[1]):
            mean = self.gaussian_points[0, gaussian_idx]
            cov = self.gaussian_covariances[gaussian_idx]

            radii, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            handle = self.server.scene.add_mesh_simple(
                name=f"{name}_{gaussian_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

            self._robot_gaussian_handles.append(handle)

    def visualize_path(
        self,
        color: tuple[int, int, int] = (50, 200, 50),
        line_width: float = 3.0,
        name: str = "Trajectory",
    ):
        """
        Visualize the task-space trajectory.

        For manipulators this is the end-effector path.
        For drones this is the center path.
        """
        task_points = []

        for q in self.trajectory:
            point = self.robot.f_task(q)

            if hasattr(point, "full"):
                point = point.full()

            task_points.append(np.asarray(point, dtype=float).reshape(3))

        task_points = np.asarray(task_points)

        if len(task_points) < 2:
            return

        segments = np.stack([task_points[:-1], task_points[1:]], axis=1)

        colors = np.tile(
            np.asarray(color, dtype=np.uint8)[None, None, :],
            (segments.shape[0], 2, 1),
        )

        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=colors,
            line_width=line_width,
        )

    def visualize_initializer_path(
        self,
        initial_trajectory: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        line_width: float = 2.0,
        name: str = "Initializer",
    ):
        initial_trajectory = np.asarray(initial_trajectory, dtype=float)

        task_points = []

        for q in initial_trajectory:
            point = self.robot.f_task(q)

            if hasattr(point, "full"):
                point = point.full()

            task_points.append(np.asarray(point, dtype=float).reshape(3))

        task_points = np.asarray(task_points)

        if len(task_points) < 2:
            return

        segments = np.stack([task_points[:-1], task_points[1:]], axis=1)

        colors = np.tile(
            np.asarray(color, dtype=np.uint8)[None, None, :],
            (segments.shape[0], 2, 1),
        )

        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=colors,
            line_width=line_width,
        )

    def visualize_gaussian_splat(
        self,
        name: str,
        means: np.ndarray,
        covariances: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
    ):
        self.server.add_gaussian_splats(
            name=name,
            centers=means,
            covariances=covariances,
            rgbs=colors,
            opacities=opacities,
        )

    def visualize_trajectory(
        self,
        dt: float = 0.1,
        loop: bool = True,
        record: bool = False,
        record_path: str | None = None,
        record_fps: float = 10.0,
    ):
        """
        Animate robot and robot Gaussians.

        If record is True, frames are grabbed from a connected browser client
        on every step and written to record_path (mp4) once you quit with
        'q' + Enter (or on any other exit) -- recording waits for a client
        to connect before the animation starts.
        """
        sample_idx = 0
        num_samples = self.trajectory.shape[0]

        frames = []
        client = None
        if record:
            print(f"Waiting for a browser client to connect before recording to {record_path} ...")
            while len(self.server.get_clients()) == 0:
                time.sleep(0.1)
            client = next(iter(self.server.get_clients().values()))
            print(f"Client connected -- recording until you quit (fps = {record_fps}).")

        try:
            while True:

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.readline().strip()
                    if key.lower() == "q":
                        print("Stopping visualization.")
                        break

                q = self.trajectory[sample_idx]

                self._update_robot(q, sample_idx)
                self._update_robot_gaussians(sample_idx)
                self._update_camera(sample_idx)

                if client is not None:
                    frames.append(client.get_render(height=720, width=1280))

                time.sleep(dt)

                sample_idx += 1

                if sample_idx >= num_samples:
                    if loop:
                        sample_idx = 0
                    else:
                        break
        finally:
            if client is not None and frames:
                iio.imwrite(record_path, frames, fps=record_fps)
                print(f"\nSaved {len(frames)} frames to {record_path}.")

    def _update_robot(self, q: np.ndarray, sample_idx: int):
        """
        Update URDF if available.

        For drones without URDF, the Gaussian visualization already shows the
        robot collision model.
        """
        if self.robot_urdf is None:
            return

        robot_type = self.robot.__class__.__name__

        if isinstance(self.robot, DroneRobot):
            qdd = self._qdd[sample_idx] if self._qdd is not None else None

            self.robot_frame.position = np.array([q[0], q[1], q[2]])
            self.robot_frame.wxyz = _rotation_matrix_to_wxyz(
                np.asarray(self.robot._body_rotation(q, qdd), dtype=float)
            )

            self.viser_urdf.update_cfg(self.urdf.zero_cfg)

            return

        elif robot_type == "MobileRobot":
            x, y, yaw = q

            self.robot_frame.position = np.array([x, y, self.robot.body_center_height])
            self.robot_frame.wxyz = np.array([
                np.cos(yaw / 2.0),
                0.0,
                0.0,
                np.sin(yaw / 2.0),
            ])

            self.viser_urdf.update_cfg(self.urdf.zero_cfg)

            return

        joint_names = getattr(self.robot, "actuated_joint_names", None)
        self.viser_urdf.update_cfg(dict(zip(joint_names, q)))

    

    def _update_robot_gaussians(self, sample_idx: int):
        if not self._robot_gaussian_handles or self._ellipsoid_factory is None:
            return

        for gaussian_idx, handle in enumerate(self._robot_gaussian_handles):
            mean = self.gaussian_points[sample_idx, gaussian_idx]
            cov = self.gaussian_covariances[gaussian_idx]

            _, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            handle.position = mean
            handle.wxyz = quat_wxyz

    def _update_camera(self, sample_idx: int):
        """
        Optionally keep the viewer camera above the robot/task point.

        Uses robot.f_task(q), so it is robot-agnostic:
        - for drones, this is usually the drone center;
        - for manipulators, this is usually the end-effector.
        """
        if not self.follow_camera:
            return

        q = self.trajectory[sample_idx]
        target = self.robot.f_task(q)

        if hasattr(target, "full"):
            target = target.full()

        target = np.asarray(target, dtype=float).reshape(3)

        camera_position = target + self.camera_offset

        look_at = target #+ np.array([0.1, 0.0, 0.1])
        next_idx = min(sample_idx + 1, self.trajectory.shape[0] - 1)

        if next_idx != sample_idx:
            next_target = self.robot.f_task(self.trajectory[next_idx])

            if hasattr(next_target, "full"):
                next_target = next_target.full()

            next_target = np.asarray(next_target, dtype=float).reshape(3)
            direction = next_target - target
            norm = np.linalg.norm(direction)

            if norm > 1e-9:
                look_at = camera_position + self.camera_lookahead * direction / norm

        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = camera_position
                client.camera.look_at = look_at

    def _create_ellipsoid_mesh(
        self,
        radii: np.ndarray,
        resolution: int = 20,
    ) -> np.ndarray:
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        u_grid, v_grid = np.meshgrid(u, v)

        x = radii[0] * np.cos(u_grid) * np.sin(v_grid)
        y = radii[1] * np.sin(u_grid) * np.sin(v_grid)
        z = radii[2] * np.cos(v_grid)

        return np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    def _create_ellipsoid_faces(
        self,
        resolution: int = 20,
    ) -> np.ndarray:
        faces = []

        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx = i * resolution + j
                faces.append([idx, idx + resolution, idx + 1])
                faces.append([idx + 1, idx + resolution, idx + resolution + 1])

        return np.asarray(faces, dtype=np.uint32)



class RobotVisualizerOnline:
    def __init__(
        self,
        robot,
        trajectory: np.ndarray,
        follow_camera: bool = False,
        camera_offset: np.ndarray | None = None,
        camera_lookahead: float = 0.5,
        duration: float | None = None,
        orbit_camera: bool = False,
        orbit_center: np.ndarray | None = None,
        orbit_radius: float = 6.0,
        orbit_height: float = 3.0,
        orbit_period: float = 20.0,
    ):
        trajectory = np.asarray(trajectory, dtype=float)

        if trajectory.ndim != 2:
            raise ValueError(
                f"trajectory must have shape (num_samples, n_dof), "
                f"got {trajectory.shape}."
            )

        if trajectory.shape[1] != robot.n_dof:
            raise ValueError(
                f"trajectory has {trajectory.shape[1]} columns, "
                f"but robot.n_dof is {robot.n_dof}."
            )

        self.robot = robot

        self.trajectory = trajectory

        # Physical duration of the trajectory, needed to scale qdd (used by
        # robots with requires_acceleration = True) into real units -- the
        # unit-index-spacing fallback (duration=None) understates or
        # overstates the tilt by whatever factor the real dt differs from 1.
        self.duration = duration
        self._qdd = (
            finite_difference_acceleration(self.trajectory, dt=self._dt())
            if getattr(robot, "requires_acceleration", False)
            else None
        )

        self.follow_camera = follow_camera
        self.camera_offset = (
            np.asarray(camera_offset, dtype=float)
            if camera_offset is not None
            else np.array([0.0, 0.0, 1.5])
        )
        self.camera_lookahead = float(camera_lookahead)

        # Orbit camera circles orbit_center at a fixed radius/height instead
        # of following the robot -- for recording a slow walkthrough of the
        # whole scene rather than a chase cam. Takes priority over
        # follow_camera when both are set.
        self.orbit_camera = orbit_camera
        self.orbit_center = (
            np.asarray(orbit_center, dtype=float)
            if orbit_center is not None
            else np.zeros(3)
        )
        self.orbit_radius = float(orbit_radius)
        self.orbit_height = float(orbit_height)
        self.orbit_period = float(orbit_period)

        self.server = ViserServer()
        self.robot_urdf = robot.urdf_path

        self._ellipsoid_faces = self._create_ellipsoid_faces()
        self._robot_gaussian_handles = []
        self._ellipsoid_factory = None

        self._obstacle_handles = []
        self._obstacle_factory = None
        self.obstacle_means = None
        self.obstacle_covariances = None

        self.gaussian_points = self._compute_robot_gaussian_points()
        self.gaussian_covariances = self._compute_robot_gaussian_covariances()

        if self.gaussian_covariances.shape[:2] != self.gaussian_points.shape[:2]:
            raise ValueError(
                "Robot Gaussian covariances must match collision points over "
                "samples and gaussian indices. "
                f"Got covariances shape {self.gaussian_covariances.shape} and "
                f"points shape {self.gaussian_points.shape}."
            )

        if self.robot_urdf is not None: 
            if not Path(self.robot_urdf).is_file():
                raise FileNotFoundError(f"URDF file not found: {self.robot_urdf}")

            self.urdf = URDF.load(self.robot_urdf)
            self.viser_urdf = ViserUrdf(self.server, urdf_or_path=self.urdf, root_node_name="/robot")

            self.robot_frame = self.server.scene.add_frame("/robot",show_axes=False)

        # if self.follow_camera:
        #     q0 = self.trajectory[0]
        #     target0 = self.robot.f_task(q0)

        #     if hasattr(target0, "full"):
        #         target0 = target0.full()

        #     target0 = np.asarray(target0, dtype=float).reshape(3)

        #     self.server.initial_camera.position = target0 + self.camera_offset
        #     self.server.initial_camera.look_at = target0
        

    def _dt(self) -> float:
        """
        Seconds between consecutive trajectory samples, for scaling qdd into
        real units. Falls back to unit index spacing (dt=1) when this
        visualizer wasn't given the trajectory's physical duration -- that
        fallback is almost always wrong by a large factor for robots with
        requires_acceleration = True, so pass duration= when visualizing one.
        """
        if self.duration is None:
            return 1.0
        return self.duration / max(self.trajectory.shape[0] - 1, 1)

    def _compute_robot_gaussian_points(self) -> np.ndarray:
        """
        Evaluate robot.collision_points(q) for all trajectory samples.

        For robots with requires_acceleration = True (e.g. a standard
        drone), qdd is a finite-difference estimate over this trajectory --
        a best-effort tilt for display, not a physically exact one.

        Returns
        -------
        np.ndarray
            Shape (num_samples, n_gaussians, 3).
        """
        all_points = []

        for t, q in enumerate(self.trajectory):
            qdd = self._qdd[t] if self._qdd is not None else None
            points = self.robot.collision_points(q, qdd)

            if hasattr(points, "full"):
                points = points.full()

            points = np.asarray(points, dtype=float).reshape(-1, 3)
            all_points.append(points)

        return np.stack(all_points, axis=0)

    def _compute_robot_gaussian_covariances(self) -> np.ndarray:
        all_covariances = []

        for t, q in enumerate(self.trajectory):
            qdd = self._qdd[t] if self._qdd is not None else None
            covs = self.robot.collision_covariances_online(q, qdd)

            if hasattr(covs, "full"):
                covs = covs.full()

            n_gaussians = self.gaussian_points.shape[1]

            covs = np.asarray(covs, dtype=float)
            covs = covs.reshape(n_gaussians, 3, 3)

            all_covariances.append(covs)

        return np.stack(all_covariances, axis=0)                

    def visualize_goal(
        self,
        goal: np.ndarray,
        radius: float = 0.05,
        color: tuple[int, int, int] = (0, 0, 255),
    ):
        self.server.scene.add_icosphere(
            name="Goal",
            position=np.asarray(goal, dtype=float),
            radius=radius,
            color=color,
        )

    def visualize_obstacles(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        n_std: float = 2.0,
        color: tuple[int, int, int] = (255, 100, 100),
        opacity: float = 0.6,
        name: str = "Obstacle",
    ):
        means = np.asarray(means, dtype=float)
        covariances = np.asarray(covariances, dtype=float)

        factory = EllipsoidFactory(n_std=float(n_std))

        for obstacle_idx, (mean, cov) in enumerate(zip(means, covariances)):
            radii, quat_wxyz = factory.cov_to_ellipsoid(cov)

            self.server.scene.add_mesh_simple(
                name=f"{name}_{obstacle_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

    def visualize_obstacles_online(
        self,
        environment,
        n_std: float = 2.0,
        color: tuple[int, int, int] = (255, 100, 100),
        opacity: float = 0.6,
        name: str = "Obstacle",
    ):
        """
        Render online/static/mobile obstacles at the first sample.
        They are updated during animation.

        Parameters
        ----------
        environment:
            GaussianEnvironmentOnline instance.

        The environment must provide:
            obstacle_means_at(k) -> shape (n_obstacles, 3)
            obstacle_covariances_at(k) -> shape (n_obstacles, 3, 3)
        """
        num_samples = self.trajectory.shape[0]

        self.obstacle_means = np.stack(
            [environment.obstacle_means_at(k) for k in range(num_samples)],
            axis=0,
        )

        self.obstacle_covariances = np.stack(
            [environment.obstacle_covariances_at(k) for k in range(num_samples)],
            axis=0,
        )

        if self.obstacle_means.ndim != 3 or self.obstacle_means.shape[2] != 3:
            raise ValueError(
                "obstacle means must have shape "
                f"(num_samples, n_obstacles, 3), got {self.obstacle_means.shape}."
            )

        if (
            self.obstacle_covariances.ndim != 4
            or self.obstacle_covariances.shape[2:] != (3, 3)
        ):
            raise ValueError(
                "obstacle covariances must have shape "
                "(num_samples, n_obstacles, 3, 3), "
                f"got {self.obstacle_covariances.shape}."
            )

        if self.obstacle_means.shape[:2] != self.obstacle_covariances.shape[:2]:
            raise ValueError(
                "Obstacle means and covariances must match over samples and "
                "obstacle indices. "
                f"Got means shape {self.obstacle_means.shape} and "
                f"covariances shape {self.obstacle_covariances.shape}."
            )

        self._obstacle_handles = []
        self._obstacle_factory = EllipsoidFactory(n_std=float(n_std))

        for obstacle_idx in range(self.obstacle_means.shape[1]):
            mean = self.obstacle_means[0, obstacle_idx]
            cov = self.obstacle_covariances[0, obstacle_idx]

            radii, quat_wxyz = self._obstacle_factory.cov_to_ellipsoid(cov)

            handle = self.server.scene.add_mesh_simple(
                name=f"{name}_{obstacle_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

            self._obstacle_handles.append(handle)

    def visualize_robot_gaussians(
        self,
        n_std: float = 2.0,
        color: tuple[int, int, int] = (80, 160, 255),
        opacity: float = 0.35,
        name: str = "RobotGaussian",
    ):
        """
        Render robot collision Gaussians at the first trajectory sample.
        They are updated during animation.
        """
        self._robot_gaussian_handles = []
        self._ellipsoid_factory = EllipsoidFactory(n_std=float(n_std))

        for gaussian_idx in range(self.gaussian_points.shape[1]):
            mean = self.gaussian_points[0, gaussian_idx]
            cov = self.gaussian_covariances[0, gaussian_idx]

            radii, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            handle = self.server.scene.add_mesh_simple(
                name=f"{name}_{gaussian_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

            self._robot_gaussian_handles.append(handle)

    def visualize_path(
        self,
        color: tuple[int, int, int] = (50, 200, 50),
        line_width: float = 3.0,
        name: str = "Trajectory",
    ):
        """
        Visualize the task-space trajectory.

        For manipulators this is the end-effector path.
        For drones this is the center path.
        """
        task_points = []

        for q in self.trajectory:
            point = self.robot.f_task(q)

            if hasattr(point, "full"):
                point = point.full()

            task_points.append(np.asarray(point, dtype=float).reshape(3))

        task_points = np.asarray(task_points)

        if len(task_points) < 2:
            return

        segments = np.stack([task_points[:-1], task_points[1:]], axis=1)

        colors = np.tile(
            np.asarray(color, dtype=np.uint8)[None, None, :],
            (segments.shape[0], 2, 1),
        )

        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=colors,
            line_width=line_width,
        )

    def visualize_initializer_path(
        self,
        initial_trajectory: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        line_width: float = 2.0,
        name: str = "Initializer",
    ):
        initial_trajectory = np.asarray(initial_trajectory, dtype=float)

        task_points = []

        for q in initial_trajectory:
            point = self.robot.f_task(q)

            if hasattr(point, "full"):
                point = point.full()

            task_points.append(np.asarray(point, dtype=float).reshape(3))

        task_points = np.asarray(task_points)

        if len(task_points) < 2:
            return

        segments = np.stack([task_points[:-1], task_points[1:]], axis=1)

        colors = np.tile(
            np.asarray(color, dtype=np.uint8)[None, None, :],
            (segments.shape[0], 2, 1),
        )

        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=colors,
            line_width=line_width,
        )

    def visualize_gaussian_splat(
        self,
        name: str,
        means: np.ndarray,
        covariances: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
    ):
        self.server.add_gaussian_splats(
            name=name,
            centers=means,
            covariances=covariances,
            rgbs=colors,
            opacities=opacities,
        )

    def visualize_trajectory(
        self,
        dt: float = 0.1,
        loop: bool = True,
        record: bool = False,
        record_path: str | None = None,
        record_fps: float = 10.0,
        record_duration: float | None = None,
    ):
        """
        Animate robot and robot Gaussians.

        If record is True, frames are grabbed from a connected browser client
        on every step and written to record_path (mp4) -- recording waits for
        a client to connect before the animation starts, then runs until you
        quit with 'q' + Enter, or automatically after record_duration seconds
        if given (useful for an orbit_camera walkthrough of a fixed length).
        """
        sample_idx = 0
        step = 0
        num_samples = self.trajectory.shape[0]

        frames = []
        client = None
        if record:
            print(f"Waiting for a browser client to connect before recording to {record_path} ...")
            while len(self.server.get_clients()) == 0:
                time.sleep(0.1)
            client = next(iter(self.server.get_clients().values()))
            print(f"Client connected -- recording until you quit (fps = {record_fps}).")

        try:
            while True:

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.readline().strip()
                    if key.lower() == "q":
                        print("Stopping visualization.")
                        break

                q = self.trajectory[sample_idx]
                elapsed_time = step * dt

                self._update_robot(q, sample_idx)
                self._update_robot_gaussians(sample_idx)
                self._update_obstacles(sample_idx)
                self._update_camera(sample_idx, elapsed_time)

                if client is not None:
                    frames.append(client.get_render(height=720, width=1280))

                if record_duration is not None and elapsed_time >= record_duration:
                    print(f"\nReached record_duration = {record_duration} s, stopping.")
                    break

                time.sleep(dt)

                step += 1
                sample_idx += 1

                if sample_idx >= num_samples:
                    if loop:
                        sample_idx = 0
                    else:
                        break
        finally:
            if client is not None and frames:
                iio.imwrite(record_path, frames, fps=record_fps)
                print(f"\nSaved {len(frames)} frames to {record_path}.")

    def _update_robot(self, q: np.ndarray, sample_idx: int):
        """
        Update URDF if available.

        For drones without URDF, the Gaussian visualization already shows the
        robot collision model.
        """
        if self.robot_urdf is None:
            return

        robot_type = self.robot.__class__.__name__

        if isinstance(self.robot, DroneRobot):
            qdd = self._qdd[sample_idx] if self._qdd is not None else None

            self.robot_frame.position = np.array([q[0], q[1], q[2]])
            self.robot_frame.wxyz = _rotation_matrix_to_wxyz(
                np.asarray(self.robot._body_rotation(q, qdd), dtype=float)
            )

            self.viser_urdf.update_cfg(self.urdf.zero_cfg)

            return

        elif robot_type == "MobileRobot":
            x, y, yaw = q

            self.robot_frame.position = np.array([x, y, self.robot.body_center_height])
            self.robot_frame.wxyz = np.array([
                np.cos(yaw / 2.0),
                0.0,
                0.0,
                np.sin(yaw / 2.0),
            ])

            self.viser_urdf.update_cfg(self.urdf.zero_cfg)

            return

        joint_names = getattr(self.robot, "actuated_joint_names", None)
        self.viser_urdf.update_cfg(dict(zip(joint_names, q)))

    def _update_robot_gaussians(self, sample_idx: int):
        if not self._robot_gaussian_handles or self._ellipsoid_factory is None:
            return

        for gaussian_idx, handle in enumerate(self._robot_gaussian_handles):
            mean = self.gaussian_points[sample_idx, gaussian_idx]
            cov = self.gaussian_covariances[sample_idx, gaussian_idx]

            _, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            handle.position = mean
            handle.wxyz = quat_wxyz

    def _update_obstacles(self, sample_idx: int):
        if (
            not self._obstacle_handles
            or self._obstacle_factory is None
            or self.obstacle_means is None
            or self.obstacle_covariances is None
        ):
            return

        for obstacle_idx, handle in enumerate(self._obstacle_handles):
            mean = self.obstacle_means[sample_idx, obstacle_idx]
            cov = self.obstacle_covariances[sample_idx, obstacle_idx]

            _, quat_wxyz = self._obstacle_factory.cov_to_ellipsoid(cov)

            handle.position = mean
            handle.wxyz = quat_wxyz

    def _update_camera(self, sample_idx: int, elapsed_time: float = 0.0):
        """
        Optionally move the viewer camera.

        orbit_camera circles orbit_center at a fixed radius/height as a
        function of elapsed_time, regardless of the robot -- for a slow,
        full walkthrough of a scene. It takes priority over follow_camera,
        which instead chases robot.f_task(q) (robot-agnostic: the drone
        center, or a manipulator's end-effector).
        """
        if self.orbit_camera:
            angle = 2.0 * np.pi * (elapsed_time / self.orbit_period)
            camera_position = self.orbit_center + np.array([
                self.orbit_radius * np.cos(angle),
                self.orbit_radius * np.sin(angle),
                self.orbit_height,
            ])

            for client in self.server.get_clients().values():
                with client.atomic():
                    client.camera.position = camera_position
                    client.camera.look_at = self.orbit_center

            return

        if not self.follow_camera:
            return

        q = self.trajectory[sample_idx]
        target = self.robot.f_task(q)

        if hasattr(target, "full"):
            target = target.full()

        target = np.asarray(target, dtype=float).reshape(3)

        camera_position = target + self.camera_offset

        look_at = target #+ np.array([0.1, 0.0, 0.1])
        next_idx = min(sample_idx + 1, self.trajectory.shape[0] - 1)

        if next_idx != sample_idx:
            next_target = self.robot.f_task(self.trajectory[next_idx])

            if hasattr(next_target, "full"):
                next_target = next_target.full()

            next_target = np.asarray(next_target, dtype=float).reshape(3)
            direction = next_target - target
            norm = np.linalg.norm(direction)

            if norm > 1e-9:
                look_at = camera_position + self.camera_lookahead * direction / norm

        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = camera_position
                client.camera.look_at = look_at

    def _create_ellipsoid_mesh(
        self,
        radii: np.ndarray,
        resolution: int = 20,
    ) -> np.ndarray:
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        u_grid, v_grid = np.meshgrid(u, v)

        x = radii[0] * np.cos(u_grid) * np.sin(v_grid)
        y = radii[1] * np.sin(u_grid) * np.sin(v_grid)
        z = radii[2] * np.cos(v_grid)

        return np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    def _create_ellipsoid_faces(
        self,
        resolution: int = 20,
    ) -> np.ndarray:
        faces = []

        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx = i * resolution + j
                faces.append([idx, idx + resolution, idx + 1])
                faces.append([idx + 1, idx + resolution, idx + resolution + 1])

        return np.asarray(faces, dtype=np.uint32)