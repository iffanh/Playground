from lagrange_polynomial import LagrangePolynomials
import numpy as np

if __name__ == '__main__':

    # Define scalars
    vt = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T
    f = np.array([0, 3, 4, 10, 7, 26])
    
    nms = LagrangePolynomials(vt, f, pdegree = 2)
    
    # print(nms.interpolate(np.array([0.5, 0.5])))