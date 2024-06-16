import numpy as np
import casadi as cas
from astar import AStar
import open3d as o3d
from gsplat_traj_optim.splines.bsplines import spline_eval

def linear_interpolation(start_point, end_point, n_control_points):
    """Return the control points of a linear interpolation between start and end point
    @args:
        start_point: np.array
        end_point: np.array
        n_control_points: int
    """
    control_points = np.zeros((n_control_points, start_point.shape[0]))
    for i in range(n_control_points):
        control_points[i] = start_point + i/(n_control_points - 1) * (end_point - start_point)
    return control_points

class BasicAStar(AStar):
    def __init__(self, bounds, occ_map, step_size = 1.0):
        self.bounds = bounds
        self.occ_map = occ_map
        self.step_size = step_size

    def neighbors(self, n):
        neighbors = []
        for dx in [-self.step_size, 0, self.step_size]:
            for dy in [-self.step_size, 0, self.step_size]:
                for dz in [-self.step_size, 0, self.step_size]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    x2 = n[0] + dx
                    y2 = n[1] + dy
                    z2 = n[2] + dz
                    if x2 < self.bounds[0][0] or x2 >= self.bounds[0][1]:
                        continue
                    if y2 < self.bounds[1][0] or y2 >= self.bounds[1][1]:
                        continue
                    if z2 < self.bounds[2][0] or z2 >= self.bounds[2][1]:
                        continue

                    if self.occ_map.check_if_included(o3d.utility.Vector3dVector(np.array([[x2, y2, z2]])))[0]:
                        continue

                    neighbors.append((x2, y2, z2))
        
        return neighbors
                    


    def distance_between(self, n1, n2):
        return np.linalg.norm(np.array(n1) - np.array(n2))
            
    def heuristic_cost_estimate(self, current, goal):
        return np.linalg.norm(np.array(current) - np.array(goal))
    
    def is_goal_reached(self, current, goal):
        return np.linalg.norm(np.array(current) - np.array(goal)) < 1
    


def astar_path_spline_fit(start_point, end_point, means, voxel_size = 1.0, num_control_points=20, z_range = None):
    min_x = min(means[:,0].min(), start_point[0], end_point[0])
    max_x = max(means[:,0].max(), start_point[0], end_point[0])
    min_y = min(means[:,1].min(), start_point[1], end_point[1])
    max_y = max(means[:,1].max(), start_point[1], end_point[1]) 
    min_z = min(means[:,2].min(), start_point[2], end_point[2])
    max_z = max(means[:,2].max(), start_point[2], end_point[2])

    start_point = start_point[:3]
    end_point = end_point[:3]

    spread_x = max_x - min_x
    spread_y = max_y - min_y
    spread_z = max_z - min_z

    bounds = [[min_x - 0.2 * spread_x, max_x + 0.2 * spread_x], [min_y - 0.2 * spread_y, max_y + 0.2 * spread_y], [min_z - 0.2 * spread_z, max_z + 0.2 * spread_z]]
    if z_range is not None:
        bounds[2] = [z_range[0], z_range[1]] 

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=voxel_size)

    astar_path_finder = BasicAStar(bounds, voxel_grid, step_size=voxel_size)

    start = (start_point[0], start_point[1], start_point[2])
    end = (end_point[0], end_point[1], end_point[2])
    path = astar_path_finder.astar(start, end)

    if path is None:
        raise ValueError("No path found")

    path = [p for p in path]

    path_arr = np.array(path)

    num_samples = len(path_arr)
    control_points = cas.SX.sym('control_points', num_control_points, 3)
    dec_vars = cas.vertcat(cas.vec(control_points))

    curve = spline_eval(control_points, num_samples)

    fitting_cost = 0
    #define the cost function
    for i in range(num_samples):
        fitting_cost += (curve[i, 0].T - path_arr[i,0])**2 + (curve[i, 1].T - path_arr[i,1])**2 + (curve[i, 2].T - path_arr[i,2])**2

    cost = fitting_cost 
    # define constraints same as cost
    cons = cas.SX([])
    lbg = []
    ubg = []
    # constrain start and end points
    for i in range(3):
        cons = cas.vertcat(cons, curve[0,i])
        lbg = np.concatenate((lbg, [path_arr[0,i] -0.005]))
        ubg = np.concatenate((ubg, [path_arr[0,i] + 0.005]))

    for i in range(3):
        cons = cas.vertcat(cons, curve[-1,i])
        lbg = np.concatenate((lbg, [path_arr[-1,i] -0.005]))
        ubg = np.concatenate((ubg, [path_arr[-1,i] +0.005]))


    ipop_options = {"ipopt.print_level": 0,
                     "ipopt.max_iter": 200, 
                     "ipopt.tol": 1e-3, 
                     "print_time": 0, 
                     "ipopt.acceptable_tol": 1e-3, 
                     "ipopt.acceptable_obj_change_tol": 1e-3, 
                     "ipopt.hessian_approximation": "limited-memory", 
                     "ipopt.mu_strategy": "adaptive", 
                     "ipopt.linear_solver": "ma27"}

    # define solver
    nlp = {'x': dec_vars, 'f': cost, 'g': cons}
    solver = cas.nlpsol('solver', 'ipopt', nlp, ipop_options)

    control_points_init = np.zeros((num_control_points, 3,1))
    sol = solver(lbg=lbg, ubg=ubg, x0=control_points_init.flatten())

    control_points_opt = np.array(sol['x']).reshape(3,num_control_points).T
    
    # a zero row to make it 4D
    control_points_opt = np.concatenate((control_points_opt, np.zeros((num_control_points,1))), axis = 1).T
    

    return control_points_opt.flatten()

