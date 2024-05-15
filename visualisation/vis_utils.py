import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


class EnvAndPathVis():
    def __init__(self):
       self.components = []

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
        pcd.paint_uniform_color(color)
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


