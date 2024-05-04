import warp as wp
from casadi import *
import numpy as np

wp.init()
wp.set_device("cuda:0")


class Convolution(Callback):
    def __init__(self, name, dim, num_points ,obstacle_means, covs_det, covs_inv,opts={}):
        Callback.__init__(self)


        assert dim == 3, "Currently only 3D is supported"

        assert len(obstacle_means) == len(covs_det) and len(covs_det) == len(covs_inv), "The number of obstacles should be the same for all input arrays"  
        assert len(obstacle_means) > 0, "The number of obstacles should be greater than 0"
        assert covs_inv.shape[1:] == (dim,dim), "The shape of the covs_inv should be (dim,dim)"

        
        
        self.dim = dim
        self.num_points = num_points
        self.num_obstacles = len(obstacle_means)

        self.obstacle_means = wp.from_numpy(obstacle_means, dtype=wp.vec3)
        self.covs_det = wp.from_numpy(covs_det, dtype=float)
        self.covs_inv = wp.from_numpy(covs_inv, dtype=wp.mat33)
        self.intermediate = wp.zeros(self.num_obstacles, dtype=float)



        self.out = wp.zeros(1, dtype=float)


        self.construct(name, opts)
        
        @wp.kernel
        def f(points: wp.array(dtype=wp.vec3),
            obstacle_means: wp.array(dtype=wp.vec3),
            covs_det: wp.array(dtype=float),
            covs_inv: wp.array(dtype=wp.mat33),
            intermediate: wp.array(dtype=wp.float32)):

            m,n = wp.tid() # m is the point index, n is the obstacle index

            diff = points[m] - obstacle_means[n]
            normal =  wp.exp(-0.5 * wp.dot(diff, covs_inv[n] @ diff)) / (wp.sqrt(2.0 * wp.pi) * covs_det[n])
            wp.atomic_add(intermediate, n, normal)
            
        self.f = f


    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self,i):
        return Sparsity.dense(self.num_points,self.dim)

    def get_sparsity_out(self,i):
        return Sparsity.dense(1,1)

    # Evaluate numerically
    def eval(self, arg):
        points= np.array(arg[0])
        self.points_gpu = wp.from_numpy(points, dtype=wp.vec3)
        self.intermediate.zero_()
        self.out.zero_()


        wp.launch(kernel = self.f,
                dim = (self.num_points, self.num_obstacles),
                inputs = [self.points_gpu, self.obstacle_means, self.covs_det, self.covs_inv,self.intermediate])

        wp.utils.array_sum(self.intermediate, out = self.out)

        
        out = (1/(self.num_obstacles * self.num_points))  * self.out.numpy()[0]
        return [out]


    def has_jacobian(self): 
        return True
    def get_jacobian(self,name,inames,onames,opts):
        

        class JacFun(Callback):
            def __init__(self, dim, num_points ,obstacle_means_gpu, covs_det_gpu, covs_inv_gpu ,opts={}):
                Callback.__init__(self)

                self.dim = dim
                self.num_points = num_points
                self.num_obstacles = len(obstacle_means_gpu)    

                self.obstacle_means = obstacle_means_gpu
                self.covs_det = covs_det_gpu
                self.covs_inv = covs_inv_gpu
                self.intermediate = wp.zeros((self.num_points, self.num_obstacles), dtype=wp.vec3)
                self.out = wp.zeros((self.num_points,1), dtype=wp.vec3)


                @wp.kernel
                def f_jac(points: wp.array(dtype=wp.vec3),
                    obstacle_means: wp.array(dtype=wp.vec3),
                    covs_det: wp.array(dtype=float),
                    covs_inv: wp.array(dtype=wp.mat33),
                    intermediate: wp.array2d(dtype=wp.vec3)):

                    m,n = wp.tid() # m is the point index, n is the obstacle index

                    diff = points[m] - obstacle_means[n]
                    normal =  wp.exp(-0.5 * wp.dot(diff, covs_inv[n] @ diff)) / (wp.sqrt(2.0 * wp.pi) * covs_det[n])

                    sub_gradient = -normal * covs_inv[n] @ diff
                    intermediate[m,n] = sub_gradient
                
                self.f_jac = f_jac

                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                n = nlpsol_out(i)
                if n == "f":
                    return Sparsity.dense(1,1)
                elif n == "x":
                    return Sparsity.dense(self.num_points,self.dim)

                else:
                    return Sparsity.dense(0,0)

            def get_sparsity_out(self,i):
                return Sparsity.dense(1,self.num_points * self.dim)


            def eval(self, arg):
                points = np.array(arg[0])
                self.points_gpu = wp.from_numpy(points, dtype=wp.vec3)
                self.intermediate.zero_()
                self.out.zero_()

                wp.launch(kernel = self.f_jac,
                        dim = (self.num_points, self.num_obstacles),
                        inputs = [self.points_gpu, self.obstacle_means, self.covs_det, self.covs_inv,self.intermediate])
                
                wp.utils.array_sum(self.intermediate, out = self.out, axis = 1)
                out = (1/(self.num_obstacles * self.num_points)) * self.out.numpy().flatten().reshape(1, self.num_points * self.dim)
                return [out]




    #         def has_jacobian(self): 
    #                 return True
    #         def get_jacobian(self,name,inames,onames,opts):
    #             print("Iname", inames)
    #             print("Onames", onames)


    #             print("Creating HessFun")
    #             class HessFun(Callback):
    #                 def __init__(self, x_gpu,opts={}):
    #                     Callback.__init__(self)
    #                     self.construct(name, opts)

    #                     self.x_gpu = x_gpu
    #                     self.dim = len(x_gpu)
    #                     self.y_gpu = wp.array([0], dtype=float)
    #                     self.out = wp.array([0], dtype=float)


    #                 def get_n_in(self): return 3
    #                 def get_n_out(self): return 2

    #                 def get_sparsity_in(self,i):
    #                     return Sparsity.dense(1,1)

    #                 def get_sparsity_out(self,i):
    #                     return Sparsity.dense(1,1)

    #                 def eval(self, arg):
    #                     out = self.dim * 2.0
    #                     return [out,0]
    #             self.jac_callback = HessFun(self.x_gpu)
    #             print("Returning HessFun")
    #             return self.jac_callback


        self.jac_callback = JacFun(self.dim, self.num_points, self.obstacle_means, self.covs_det, self.covs_inv)
        return self.jac_callback




if __name__ == "__main__":
    # Define the problem
    dim = 3
    num_obstacles = 1_000_000
    num_points = 30

    robot_cov = np.eye(dim)
    covs = np.array([np.eye(dim) for _ in range(num_obstacles)])
    covs_sum = covs + robot_cov


    covs_inv = covs.copy()
    covs_det = np.ones(num_obstacles)

    obstacle_means = np.ones((num_obstacles, dim))
    points = np.zeros((num_points, dim)) 

    conv = Convolution("conv", dim, num_points, obstacle_means, covs_det, covs_inv)


    y = MX.sym("y", num_points, dim)
    jac = Function("jac_conv", [y], [jacobian(conv(y), y)])

    j = jac(points)


    import timeit
    num_tests = 100
    time = timeit.timeit(lambda: jac(points), number=num_tests)
    print(time/num_tests)


        

