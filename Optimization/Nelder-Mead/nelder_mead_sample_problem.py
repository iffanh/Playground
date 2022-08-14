from nelder_mead_solver import NelderMeadSolver

import numpy as np

def objfunc(vec):
    """
    A function that maps the input vector to objective funcdtion scalar
    """
    conjArr = np.array([[1, 0.2, 1.3, 1], [0.2, 2, 0.1, 1.4], [1, 0.4, 2.3, 1], [1.5, 2, 2.3, 1]])

    scalar = np.dot(np.dot(vec.T, conjArr), vec)

    return scalar

if __name__ == '__main__':

    # Define scalars
    arr = np.array([[1, 2, 1, 0], [0, 0, 1, 3], [1, 1, 0, 0.2], [2, 1, 1, 3]])

    nms = NelderMeadSolver(arr, objfunc, alpha=0.2, tol=10E-7, maxIter=100)
    nms.solve()
    print(nms.solution)