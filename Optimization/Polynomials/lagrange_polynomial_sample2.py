from lagrange_polynomial import LagrangePolynomials
import numpy as np

if __name__ == '__main__':
    
    ### This follows from Figure 3.1 - 3.4 in Conn's book. 
    ### Notice the differnece in the polynomial degree. Here we are using polynomial of degree 2 instead of 1 in the book
    
    print(f"==============================================")
    vt = np.array([[0.05, 0.1], [0.1, 0.05], [0.5, 0.5], [0.95, 0.9], [0.9, 0.95], [0.85, 0.85]]).T
    f = np.array([0, 3, 4, 10, 7, 26])

    nms3 = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Poisedness value found: {nms3.poisedness()} ")
    for i, p in enumerate(nms3.lagrange_polynomials):
            print(f"l_{i} : {p.symbol}")

    print(f"==============================================")
    vt = np.array([[0.01, 0.02], [0.02, 0.01], [0.5, 0.5], [0.99, 0.98], [0.98, 0.98], [0.97, 0.97]]).T
    f = np.array([0, 3, 4, 10, 7, 26])

    nms4 = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Poisedness value found: {nms4.poisedness()} ")
    for i, p in enumerate(nms4.lagrange_polynomials):
            print(f"l_{i} : {p.symbol}")

    print(f"==============================================")
    vt = np.array([[0.524, 0.0006], [0.032, 0.323], [0.187, 0.890], [0.854, 0.853], [0.982, 0.368], [0.774, 0.918]]).T
    f = np.array([0, 3, 4, 10, 7, 26])

    nms5 = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Poisedness value found: {nms5.poisedness()} ")
    for i, p in enumerate(nms5.lagrange_polynomials):
            print(f"l_{i} : {p.symbol}")

    print(f"==============================================")
    vt = np.array([[0.524, 0.0006], [0.032, 0.323], [0.187, 0.890], [0.5, 0.5], [0.982, 0.368], [0.774, 0.918]]).T
    f = np.array([0, 3, 4, 10, 7, 26])

    nms6 = LagrangePolynomials(vt, f, pdegree = 2)
    print(f"Poisedness value found: {nms6.poisedness()} ")
    for i, p in enumerate(nms6.lagrange_polynomials):
            print(f"l_{i} : {p.symbol}")