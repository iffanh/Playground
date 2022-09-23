from lagrange_polynomial import LagrangePolynomials
import numpy as np

if __name__ == '__main__':

    print(f"==============================================")
    # Define scalars
    vt = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T
    f = np.array([0, 3, 4, 10, 7, 26])
    
    nms = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Interpolation value at point (0.5, 0.5) = {nms.interpolate(np.array([0.5, 0.5]))}")
    print(f"Poisedness value found: {nms.poisedness()} ")
    print(f"\nLagrange polynomials are:")
    for i, p in enumerate(nms.lagrange_polynomials):
        print(f"l_{i} : {p.symbol}")
        print(f"l_{i}({nms.v[:, i]}) : {p.feval(*nms.v[:, i])}")
    
    print(f"==============================================")
    # Exercise 3 From Chapter 3 in Conn's book: Introduction to derivative-free optimization
    vt = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [1, 1], [0, -1]]).T
    f = np.array([0, 3, 4, 10, 7, 26])
    
    nms = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Interpolation value at point (0.5, 0.5) = {nms.interpolate(np.array([0.5, 0.5]))}")
    print(f"Poisedness value found: {nms.poisedness()} ")
    for i, p in enumerate(nms.lagrange_polynomials):
        print(f"l_{i} : {p.symbol}")
        print(f"l_{i}({nms.v[:, i]}) : {p.feval(*nms.v[:, i])}")
    
    print(f"==============================================")
    # Page. 61 From Chapter 4 in Conn's book: Introduction to derivative-free optimization
    vt = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [0.5, 0.5]]).T
    f = np.array([0, 3, 4, 10, 7, 26, -2])
    
    nms = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Interpolation value at point (0.5, 0.5) = {nms.interpolate(np.array([0.5, 0.5]))}")
    print(f"Poisedness value found: {nms.poisedness()} ")
    for i, p in enumerate(nms.lagrange_polynomials):
        print(f"l_{i} : {p.symbol}")
        print(f"l_{i}({nms.v[:, i]}) : {p.feval(*nms.v[:, i])}")
        
