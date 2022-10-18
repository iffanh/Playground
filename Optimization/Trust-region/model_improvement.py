from lagrange_polynomial import LagrangePolynomials
from lagrange_polynomial import ModelImprovement
import numpy as np
from sympy import poly

import matplotlib.pyplot as plt

def myfunc(x:np.ndarray) -> np.ndarray:
    return x[0] + x[1] + 2*x[0]**2 + 3*x[1]**3

if __name__ == '__main__':

    print(f"==============================================")
    # Define scalars

    dataset = np.array([[0.524, 0.0006], [0.032, 0.323], [0.187, 0.890], [0.854, 0.853], [0.982, 0.368], [0.774, 0.918]]).T # nearly lie in a circle
    # dataset = np.array([[0.05, 0.1], [0.1, 0.05], [0.5, 0.5], [0.95, 0.9], [0.9, 0.95], [0.85, 0.85]]).T
    # dataset = np.array([[0.01, 0.02], [0.02, 0.01], [0.5, 0.5], [0.99, 0.98], [0.98, 0.98], [0.97, 0.97]]).T
    # dataset = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0], [0.0, 2.0]]).T
    # dataset = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0], [0.0, -1.0]]).T
    # dataset = np.array([[0.524, 0.0006], [0.032, 0.323], [0.187, 0.890], [0.5, 0.5], [0.982, 0.368], [0.774, 0.918]]).T
    results = []
    for i in range(dataset.shape[1]):
        x = dataset[:, i]
        results.append(myfunc(x))
    results = np.array(results)

    
    lp = LagrangePolynomials(dataset, results, pdegree = 2, sort_type='function')
    mi = ModelImprovement()
    lp2 = mi.improve_model(lpolynomials=lp, func=myfunc, L=1.5, max_iter=10, sort_type='function')
    
    print(f"old polynomials = {lp.model_polynomial.symbol}")
    print(f"old poisedness = {lp.poisedness().max_poisedness()}")
    print(f"new polynomials = {lp2.model_polynomial.symbol}")
    print(f"new poisedness = {lp2.poisedness().max_poisedness()}")
    
    plt.figure()
    plt.scatter(dataset[0, :], dataset[1, :], color='green',  marker='*')
    plt.scatter(lp.y[0, :], lp.y[1, :], color='red',  marker='x')
    plt.scatter(lp2.y[0, :], lp2.y[1, :], color='blue')
    plt.savefig(f"./data/model_improvement.png")