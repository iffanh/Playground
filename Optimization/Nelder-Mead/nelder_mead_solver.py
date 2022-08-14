import numpy as np

"""
This is an attempt to create a nelder-mead solver for a non-linear optimization
problem

"""
def check_input_validity(arr:np.ndarray, f:callable):

    for col in range(arr.shape[1]):

        vec = arr[:,col]
        try:
            f(vec)
        except:
            raise Exception("Array and the map function is not compatible")

def order_by_objective_function(arr:np.ndarray, f:callable):

    sorted_f, sorted_col = zip(*sorted([(f(arr[:,col]), col) for col in range(arr.shape[1])], key=lambda x:x[0]))
    print(sorted_f, sorted_col)
    arr = arr[:, sorted_col]
    return arr

def calculate_centroid(arr:np.ndarray):
    centroid = np.mean(arr[:,:-1], axis=1)
    return centroid

def calculate_reflected_point(centroid, alpha, vec):
    return centroid + alpha*(centroid - vec)

def calculate_expanded_point(centroid, gamma, r_point):
    return centroid + gamma*(r_point - centroid)

def calculate_contracted_point(centroid, rho, r_point):
    return centroid + rho*(r_point - centroid)

def replace_all_points(arr:np.ndarray, omega):
    arr[:, 1:] = arr[:,[0]] + omega*(arr[:, 1:] - arr[:, [0]])
    return arr

def run_nelder_mead(arr:np.ndarray, f:callable, alpha:float=1.0, gamma:float=2.0, rho:float=0.5, omega=0.5, tol:float=10E-5, maxIter:int=100):

    """
    arr  : an array of input (state) where each column represents a point
    f   : a function that maps to the objective function that we would like to minimize
    tol : maximum tolerance to
    """

    check_input_validity(arr, f)

    for i in range(maxIter):

        print("------- Iteration %s -------" %i)
        print(arr)
        arr = order_by_objective_function(arr, f)
        centroid = calculate_centroid(arr)
        r_point = calculate_reflected_point(centroid, alpha, arr[:, -1])
        print(f(r_point))
        if f(r_point) >= f(arr[:, 0]) and f(r_point) < f(arr[:, -2]):
            arr[:, -1] = r_point

        elif f(r_point) < f(arr[:,0]):
            e_point = calculate_expanded_point(centroid, gamma, r_point)
            if f(e_point) < f(r_point):
                arr[:, -1] = e_point
            else:
                arr[:, -1] = r_point

        else:
            if f(r_point) < f(arr[:, -1]):
                c_point = calculate_contracted_point(centroid, rho, r_point)
                if f(c_point) < f(r_point):
                    arr[:, -1] = c_point
                else:
                    arr = replace_all_points(arr, omega)

            elif f(r_point) >= f(arr[:,-1]):
                c_point = calculate_contracted_point(centroid, rho, arr[:, -1])
                if f(c_point) < f(arr[:, -1]):
                    arr[:, -1] = c_point
                else:
                    arr = replace_all_points(arr, omega)
    return arr

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

    run_nelder_mead(arr, objfunc, maxIter=200)