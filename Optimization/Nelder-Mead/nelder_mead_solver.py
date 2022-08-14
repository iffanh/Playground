import numpy as np

"""
This is an attempt to create a nelder-mead solver for a non-linear optimization
problem

"""

class NelderMeadSolver:
    def __init__(self, arr:np.ndarray=None, f:callable=None, alpha:float=1.0, gamma:float=2.0, rho:float=0.5, omega=0.5, tol:float=10E-5, maxIter:int=100) -> None:

        self.arr = arr
        self.f = f
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.omega = omega
        self.tol = tol
        self.maxIter = maxIter

        self.sol_arr = None

        pass

    def check_input_validity(self, arr:np.ndarray, f:callable):

        for col in range(arr.shape[1]):

            vec = arr[:,col]
            try:
                f(vec)
            except:
                raise Exception("Array and the map function is not compatible")

    def order_by_objective_function(self, arr:np.ndarray, f:callable):

        _, sorted_col = zip(*sorted([(f(arr[:,col]), col) for col in range(arr.shape[1])], key=lambda x:x[0]))
        arr = arr[:, sorted_col]
        return arr

    def calculate_centroid(self, arr:np.ndarray):
        centroid = np.mean(arr[:,:-1], axis=1)
        return centroid

    def calculate_reflected_point(self, centroid, alpha, vec):
        return centroid + alpha*(centroid - vec)

    def calculate_expanded_point(self, centroid, gamma, r_point):
        return centroid + gamma*(r_point - centroid)

    def calculate_contracted_point(self, centroid, rho, r_point):
        return centroid + rho*(r_point - centroid)

    def replace_all_points(self, arr:np.ndarray, omega):
        arr[:, 1:] = arr[:,[0]] + omega*(arr[:, 1:] - arr[:, [0]])
        return arr

    def calculate_standard_error(self, arr:np.ndarray, f:callable):

        obj_values = [f(arr[:,col]) for col in range(arr.shape[1])]
        mean_obj_values = np.mean(obj_values)

        test = np.sqrt(np.sum([(ov-mean_obj_values)**2 for ov in obj_values])/(arr.shape[1] - 1))
        return test

    def solve(self):

        """
        arr  : an array of input (state) where each column represents a point
        f   : a function that maps to the objective function that we would like to minimize
        tol : maximum tolerance to
        """

        self.check_input_validity(self.arr, self.f)

        for i in range(self.maxIter):

            print("------- Iteration %s -------" %i)
            self.arr = self.order_by_objective_function(self.arr, self.f)
            stderr = self.calculate_standard_error(self.arr, self.f)
            print("Best : %s" %self.f(self.arr[:,0]))
            print("StandardError : %s" %stderr)

            if stderr < self.tol:
                self.sol_arr = self.arr
                break

            centroid = self.calculate_centroid(self.arr)
            r_point = self.calculate_reflected_point(centroid, self.alpha, self.arr[:, -1])
            if self.f(r_point) >= self.f(self.arr[:, 0]) and self.f(r_point) < self.f(self.arr[:, -2]):
                self.arr[:, -1] = r_point

            elif self.f(r_point) < self.f(self.arr[:,0]):
                e_point = self.calculate_expanded_point(centroid, self.gamma, r_point)
                if self.f(e_point) < self.f(r_point):
                    self.arr[:, -1] = e_point
                else:
                    self.arr[:, -1] = r_point

            else:
                if self.f(r_point) < self.f(self.arr[:, -1]):
                    c_point = self.calculate_contracted_point(centroid, self.rho, r_point)
                    if self.f(c_point) < self.f(r_point):
                        self.arr[:, -1] = c_point
                    else:
                        arr = self.replace_all_points(self.arr, self.omega)

                elif self.f(r_point) >= self.f(self.arr[:,-1]):
                    c_point = self.calculate_contracted_point(centroid, self.rho, self.arr[:, -1])
                    if self.f(c_point) < self.f(self.arr[:, -1]):
                        self.arr[:, -1] = c_point
                    else:
                        self.arr = self.replace_all_points(self.arr, self.omega)

        self.sol_arr = self.arr

def objfunc(vec):
    """
    A function that maps the input vector to objective funcdtion scalar
    """
    conjArr = np.array([[1, -0.2, 3], [2, 1, -1.4], [4, -2.3, 5]])

    scalar = np.dot(np.dot(vec.T, conjArr), vec)

    return scalar

if __name__ == '__main__':

    # Define scalars
    arr = np.array([[1, 2, 1], [0, 0, 1], [-1, 1, 0]])

    nms = NelderMeadSolver(arr, objfunc, maxIter=200)
    nms.solve()
    print(nms.sol_arr)