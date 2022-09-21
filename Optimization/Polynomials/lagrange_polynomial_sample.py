from lagrange_polynomial import LagrangePolynomials
import numpy as np

if __name__ == '__main__':

    print(f"==============================================")
    # Define scalars
    vt = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T
    f = np.array([0, 3, 4, 10, 7, 26])
    
    nms1 = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Interpolation value at point (0.5, 0.5) = {nms1.interpolate(np.array([0.5, 0.5]))}")
    print(f"Poisedness value found: {nms1.poisedness()} ")
    print(f"\nLagrange polynomials are:")
    for i, p in enumerate(nms1.lagrange_polynomials):
        print(f"l_{i} : {p.symbol}")
    
    print(f"==============================================")
    # Exercise 3 From Chapter 3 in Conn's book: Introduction to derivative-free optimization
    vt = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [1, 1], [0, -1]]).T
    f = np.array([0, 3, 4, 10, 7, 26])
    
    nms2 = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Interpolation value at point (0.5, 0.5) = {nms2.interpolate(np.array([0.5, 0.5]))}")
    print(f"Poisedness value found: {nms2.poisedness()} ")
    for i, p in enumerate(nms2.lagrange_polynomials):
        print(f"l_{i} : {p.symbol}")
    
