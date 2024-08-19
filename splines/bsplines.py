import matplotlib
import numpy as np
import os
import logging
import casadi as cas

from scipy.spatial import ConvexHull

logging.basicConfig(level =logging.INFO)


# ==================================== FEATURE MATRICIES ====================================

MINVO_3 = np.array([
    [-3.4416309793565660335445954842726,  6.9895482693324069156659561485867, -4.4622887879670974919932291413716,                  0.91437149799125659734933338484986],
    [ 6.6792587678886103930153694818728, -11.845989952130473454872117144987,  5.2523596862506065630071816485724, -0.000000000000000055511151231257827021181583404541],
    [-6.6792587678886103930153694818728,  8.1917863515353577241739913006313, -1.5981560856554908323090558042168,                 0.085628502008743445639282754200394],
    [ 3.4416309793565660335445954842726,  -3.335344668737291184967830304231, 0.80808518737198176129510329701588, -0.000000000000000012522535092207576212786079850048],
]).T

MINVO_2 = np.array([
    [ 1.4999999992328318931811281800037, -2.3660254034601951866889635311964,   0.9330127021136816189983420599674],
    [-2.9999999984656637863622563600074,  2.9999999984656637863622563600074,                                   0],  
    [ 1.4999999992328318931811281800037, -0.6339745950054685996732928288111, 0.066987297886318325490506708774774],
]).T

MINVO_1 = np.array([
    [-1, 1],
    [1,0],
]).T

MINVO_0 = np.array([
    [1],
]).T

BSPLINE_3 = np.array([
    [-0.16666666666666666666666666666667,  0.5, -0.5, 0.16666666666666666666666666666667],
    [                                0.5, -1.0,    0, 0.66666666666666666666666666666667],
    [                               -0.5,  0.5,  0.5, 0.16666666666666666666666666666667],
    [ 0.16666666666666666666666666666667,    0,    0,                                  0]

]).T

BSPLINE_2 = np.array([
    [ 0.5, -1.0, 0.5],
    [-1.0,  1.0, 0.5],
    [ 0.5,    0,   0]
]).T 


B_SPLINE_1 = np.array([ 
    [-1, 1],
    [1, 0]
]).T

B_SPLINE_0 = np.array([
    [1]
]).T


def basis_function(t, derivate = 0):
    assert derivate in [0,1,2,3], "derivate must be 0, 1 or 2"

    if derivate == 0:
      t_ = np.array([t**3, t**2, t, 1])
    if derivate == 1:
       t_ = np.array([3 *t**2, 2*t, 1,0])
    if derivate == 2:
       t_ = np.array([6*t,2, 0, 0])
    if derivate == 3:
        t_ = np.array([6,0, 0, 0])

    return t_ @ BSPLINE_3

def basis_function_mat(ts, n_control_points, n_knots, derivate = 0):
    mat = np.zeros((ts.shape[0], n_control_points))
    for i, t in enumerate(ts):
        offset = max(min(int(np.floor(t)), n_knots -1),0)
        u = t - offset
        mat[i][offset: offset + 4] = basis_function(u, derivate = derivate)
    
    return mat

def spline_eval(control_points, num_samples, derivate = 0):
    n_knots = control_points.shape[0] - 4

    ts = np.linspace(0, n_knots, num_samples)
    basis = basis_function_mat(ts, control_points.shape[0], n_knots, derivate = derivate)
    curve = basis @ control_points

    return curve


def spline_eval_at_s(control_points, s, derivate = 0):
    n_knots = control_points.shape[0] - 4

    ts = np.array([s])
    basis = basis_function_mat(ts, control_points.shape[0], n_knots, derivate = derivate)
    curve = basis @ control_points
    
    point = curve[0]

    return point




def get_minvo_hulls(control_points, derivative = 0):
    assert derivative in [0,1,2,3,], "higher order derivatives not supported"

    control_points_type = type(control_points).__name__


    V = control_points
    dV = cas.MX.zeros(control_points.shape[0]-1, control_points.shape[1]) if control_points_type == "MX" else np.zeros((control_points.shape[0]-1, control_points.shape[1]))
    ddV = cas.MX.zeros(control_points.shape[0]-2, control_points.shape[1]) if control_points_type == "MX" else np.zeros((control_points.shape[0]-2, control_points.shape[1]))
    dddV = cas.MX.zeros(control_points.shape[0]-3, control_points.shape[1]) if control_points_type == "MX" else np.zeros((control_points.shape[0]-3, control_points.shape[1]))


    if derivative >= 1:
        for i in range(control_points.shape[0]-1):
            dV[i,:] =  (control_points[i +1,:] - control_points[i,:])
    if derivative >= 2:
        for i in range(control_points.shape[0]-2):
            ddV[i,:] =  (dV[i +1,:] - dV[i,:])
    if derivative >= 3:
        for i in range(control_points.shape[0]-3):
            dddV[i,:] =  (ddV[i +1,:] - ddV[i, :])

    hulls = [] 
    if derivative == 0:
        for i in range(control_points.shape[0]-3):
            hull = np.linalg.inv(MINVO_3) @ BSPLINE_3 @ V[i:i+4,:]
            # hull = V[i:i+4,:]
            hulls.append(hull)

    elif derivative == 1:
        for i in range(control_points.shape[0]-4):
            hull = np.linalg.inv(MINVO_2) @ BSPLINE_2 @ dV[i:i+3,:]
            # hull = dV[i:i+3,:]
            hulls.append(hull)

    elif derivative == 2:
        for i in range(control_points.shape[0]-5):
            hull = np.linalg.inv(MINVO_1) @ B_SPLINE_1 @ ddV[i:i+2,:]
            # hull = ddV[i:i+2,:]
            hulls.append(hull)

    elif derivative == 3:
        for i in range(control_points.shape[0]-6):
            hull = np.linalg.inv(MINVO_0) @ B_SPLINE_0 @ dddV[i:i+1,:]
            # hull = dddV[i:i+1,:]
            hulls.append(hull)

    return hulls




if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    n_control_points = 10
    n_knots = n_control_points - 4

    control_points = np.array(
        [ 
            [0,0],
            [1,3],
            [4,0],
            [7,-3],
            [11,0],
            [15,3],
            [20,0],
            [27,-3],
            [40,0],
            [50,3] ,
        ]
    )

    curve = spline_eval(control_points, 1000)
    dcurve = spline_eval(control_points, 1000, derivate = 1)
    ddcurve = spline_eval(control_points, 1000, derivate = 2)
    dddcurve = spline_eval(control_points, 1000, derivate = 3)


    points = []
    vels = []
    accs = []
    for s in np.linspace(0, n_knots, 1000):

        point = spline_eval_at_s(control_points, s)
        vel = spline_eval_at_s(control_points, s, derivate = 1)
        acc = spline_eval_at_s(control_points, s, derivate = 2)

        points.append(point)

        vels.append(vel)
        accs.append(acc)


    S = n_knots
    T = 5

    m = S/T

    s = np.linspace(0, S, 1000) 
    ts = s/m


  
    plt.plot(ts, curve[:,0], label = "position_x")
    plt.plot(ts, curve[:,1], label = "position_y")

    plt.plot(ts, m *dcurve[:,0], label = "velocity_x")
    plt.plot(ts, m *dcurve[:,1], label = "velocity_y")

    plt.plot(ts, m**2 * ddcurve[:,0], label = "acceleration_x")
    plt.plot(ts, m**2 *ddcurve[:,1], label = "acceleration_y")

    plt.plot(ts, m**3 * dddcurve[:,0], label = "jerk_x")
    plt.plot(ts, m**3 * dddcurve[:,1], label = "jerk_y")


    
    # plt.plot(dcurve[:,0], dcurve[:,1], label = "velocity")
    # plt.plot(ddcurve[:,0], ddcurve[:,1], label = "acceleration")

    plt.legend()

    plt.show()  

    
    hulls = get_minvo_hulls(control_points, derivative = 0)
    plt.axis('equal')
    for hull in hulls:
        plt.scatter(hull[:,0], hull[:,1])
        try:
            indices = ConvexHull(hull)
            for simplex in indices.simplices:
                plt.plot(hull[simplex,0], hull[simplex,1], label = "hull")
        except:
            pass

    plt.plot(curve[:,0], curve[:,1], "--", label = "position", color = "red")
    plt.show()

    hulls = get_minvo_hulls(control_points, derivative = 1)
    plt.axis('equal')
    for hull in hulls:
        plt.scatter(hull[:,0], hull[:,1])
        plt.plot(hull[:,0], hull[:,1])
        try:
            indices = ConvexHull(hull)
            for simplex in indices.simplices:
                plt.plot(hull[simplex,0], hull[simplex,1], label = "hull")
        except:
            pass




    plt.plot(dcurve[:,0], dcurve[:,1], "--", label = "position", color = "red")
    plt.show()



    hulls = get_minvo_hulls(control_points, derivative = 2)
    plt.axis('equal')
    for hull in hulls:
        plt.scatter(hull[:,0], hull[:,1],color ="blue")
        plt.plot(hull[:,0], hull[:,1])
        try:
            indices = ConvexHull(hull)
            for simplex in indices.simplices:
                plt.plot(hull[simplex,0], hull[simplex,1], label = "hull")
        except:
            pass

    plt.plot(ddcurve[:,0], ddcurve[:,1], "--", label = "position", color = "red")
    plt.show()


    hulls = get_minvo_hulls(control_points, derivative = 3)

    

    plt.axis('equal')
    for hull in hulls:
        plt.scatter(hull[:,0], hull[:,1], color = "blue")
        plt.plot(hull[:,0], hull[:,1])
        try:
            indices = ConvexHull(hull)
            for simplex in indices.simplices:
                plt.plot(hull[simplex,0], hull[simplex,1], label = "hull")
        except:
            pass

    plt.plot(dddcurve[:,0], dddcurve[:,1], "--", label = "position", color = "red")
    plt.show()


    