import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys
import viser
import time

class BaseVis():
    def __init__(self):
        self.components = []

    def add_points(self, points, color = [0,0,1]):
        """Add points to the visualization
        @args:
            points: np.array
            color: list
        """
        raise NotImplementedError

    def add_curve(self, points, color = [1,0,0]):
        """Add curve to the visualization
        @args:
            points: np.array
            color: list
        """
        raise NotImplementedError

    def add_gaussians(self, means, covs, color = [0,1,0]):
        """Add gaussians to the visualization
        @args:
            means: np.array
            covs: np.array
            color: list
        """
        raise NotImplementedError

    def add_gaussian_path(self, curve, cov, kinematics,color = [0,1,0]):
        """Add gaussians to the visualization
        @args:
            curve: np.array
            cov: np.array
            kinematics: object
            color: list
        """
        raise NotImplementedError

    def show(self):
        """Show the visualization
        """
        raise NotImplementedError

    def z_colormap(self, points):
        """Generate colors for points based on z value"""
        z = points[:,2]
        z = (z - np.min(z)) / (np.max(z) - np.min(z))
        cmap = plt.get_cmap("viridis")
        colors = cmap(z)
        return colors[:,:3]

    def lin_colormap(self, num):
        """Generate linear colors"""
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, num))
        return colors[:,:3]


    def path_colormap(self, num, kin):
        """Generate colors that based on kinematic gaussians"""
        robot = np.linspace(0, 1, kin)
        cmap = plt.get_cmap("brg")


        path = np.tile(robot, num)
        colors = cmap(path)
        return colors[:,:3]


class ViserVis(BaseVis):
    def __init__(self):
        self.server = viser.ViserServer()
        self.running = True
        self.valid = True

    def add_points(self, points, color = [0,0,1]):
        self.server.scene.add_point_cloud("Points", points, color)

    def add_curve(self, points, color = [1,0,0]):

        self.server.scene.add_spline_catmull_rom("Curve", positions=points, color=color)

    def add_gaussian_path(self, curve, cov, kinematics, color = [0,1,0]):
        kinematics_functor = kinematics.map(curve.shape[0], "openmp")
        means = np.array(kinematics_functor(curve.T).T)

        num, kin = curve.shape

        rgb = self.path_colormap(num, kin-1)
        opacity = np.ones((len(means), 1))

        covs = np.tile(cov, (len(means), 1, 1))

        self.server.add_gaussian_splats("Robot Gaussians", means, covs, rgb, opacity)

    def add_gaussians(self, means, covs, color = [0,1,0], opacity = 1.0):
        if  len(color) == 3:
            color = self.z_colormap(means)
        if type(opacity) == float:
            opacity = np.tile(opacity, (len(means), 1))

        means = np.ascontiguousarray(means)
        
        print(f"means: {means.shape}, covs: {covs.shape}, color: {color.shape}, opacity: {opacity.shape}")
        self.server.add_gaussian_splats("Scene Splat", means, covs, color, opacity, visible=False)


    def show(self):
        while self.running:
            time.sleep(0.1)

        print(f"Closing viewer with status {self.valid}")

        return self.valid


class EnvAndPathVis(BaseVis):

    def add_voxels(self, points, voxel_size = 1):
        """Add voxels to the visualization
        @args:
            points: np.array
            voxel_size: float
            color: list
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                        voxel_size=voxel_size)

        self.components.append(voxel_grid)


    def add_points(self, points, color = [0,0,1]):
        """Add points to the visualization
        @args:
            points: np.array
            color: list
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.paint_uniform_color(color)
        self.components.append(pcd)

    def add_curve(self, points, color = [1,0,0]):
        """Add curve to the visualization
        @args:
            points: np.array
            color: list
        """
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = []
        for i in range(len(points) - 1):
            lines.append([i, i+1])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        self.components.append(line_set)

    def add_gaussians(self, means, covs, color = [0,1,0]):
        """Add gaussians to the visualization
        @args:
            means: np.array
            covs: np.array
            color: list
        """
        # draw ellipsoids 
        for i in range(means.shape[0]):
            # compute max eigenvalue
            eigval, eigvec = np.linalg.eig(covs[i].reshape(3,3))
            max_eigval = np.max(eigval)
            radius = np.sqrt(max_eigval)

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(color)

            # Define translation matrix
            translation = np.identity(4)
            translation[0:3, 3] = means[i,:]

            # Apply translation to the sphere
            sphere.transform(translation)
            self.components.append(sphere)
                    
    def add_gaussian_path(self, curve, cov, kinematics,color = [0,1,0]):
        kinematics_functor = kinematics.map(curve.shape[0], "openmp")
        means = np.array(kinematics_functor(curve.T).T)

        self.add_curve(curve[:,:3], color)
        eigval, eigvec = np.linalg.eig(cov.reshape(3,3))
        max_eigval = np.max(eigval)
        radius = np.sqrt(max_eigval)
        
        for i in range(len(means)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(color)

            # Define translation matrix
            translation = np.identity(4)
            translation[0:3, 3] = means[i,:]

            # Apply translation to the sphere
            sphere.transform(translation)
            self.components.append(sphere)
            

    def show(self):
        """Show the visualization
        """
        o3d.visualization.draw_geometries(self.components)




if __name__ == "__main__":
    # if -l flag is passed, use the legacy viewer
    if "-l" in sys.argv:
        vis = EnvAndPathVis()
        print("Using legacy open3d viewer")
    else:
        vis = ViserVis()
        print("Using viser viewer")

    obs_pos = np.load("test_data/obstacle_positions.npy", allow_pickle=True)
    obs_cov = np.load("test_data/obstacle_covs.npy", allow_pickle=True)
    spline = np.load("test_data/spline.npy", allow_pickle=True)
    opt_curve = np.load("test_data/opt_curve.npy", allow_pickle=True)
    robot_cov = np.load("test_data/robot_cov.npy", allow_pickle=True)
    kinematics = np.load("test_data/kinematics.npy", allow_pickle=True).item()
    means = np.load("test_data/means.npy", allow_pickle=True)
    covs = np.load("test_data/covs.npy", allow_pickle=True)

    # make obs pos contiguous
    obs_pos = np.ascontiguousarray(obs_pos)





    vis.add_points(obs_pos, color = [0,0,1])
    vis.add_curve(spline, color = [1,0,0])
    vis.add_gaussian_path(opt_curve, robot_cov, kinematics, color = [0,1,0])
    vis.add_gaussians(obs_pos, obs_cov, color = [0,1,0])
    vis.show()