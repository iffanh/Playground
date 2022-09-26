from lagrange_polynomial import LagrangePolynomials
from generate_poised_sets import Ball
import numpy as np
import matplotlib.pyplot as plt

def myfunc(x:np.ndarray) -> np.ndarray:
    return x[0]**3 - 2*x[1]**3 - 3*(x[0]**2) - x[1]



if __name__ == '__main__':
    
    print(f"==============================================")
    
    levels = [-10, -5, -2, -1, 1, 2, 5, 10]
    pdegree = 1
    
    # vt = np.array([[0.524, 0.0006], [0.032, 0.323], [0.187, 0.890], [0.854, 0.853], [0.982, 0.368], [0.774, 0.918]]).T
    # vt = np.array([[0.01, 0.02], [0.02, 0.01], [0.5, 0.5], [0.99, 0.98], [0.98, 0.98], [0.97, 0.97]]).T
    vt = np.array([[0.05, 0.1], [0.1, 0.05], [0.5, 0.5], [0.95, 0.9], [0.9, 0.95], [0.85, 0.85]]).T
    # vt = np.array([[0.524, 0.0006], [0.032, 0.323], [0.187, 0.890], [0.5, 0.5], [0.982, 0.368], [0.774, 0.918]]).T
    results = []
    for i in range(vt.shape[1]):
        x = vt[:, i]
        results.append(myfunc(x))
    results = np.array(results)

    nms3 = LagrangePolynomials(vt, results, pdegree = pdegree)
    print(f"Poisedness value found: {nms3.poisedness()} ")
    for i, p in enumerate(nms3.lagrange_polynomials):
            print(f"l_{i} : {p.symbol}")

    x, y = np.meshgrid(np.linspace(-2, 2, 100),
                    np.linspace(-2, 2, 100))

    # Functions
    fig, ax = plt.subplots(1)
    ax.contour(x, y, myfunc([x, y]), levels)
    plt.scatter(vt[0, :], vt[1, :])
    
    filename = './plot_original_function.png'
    plt.savefig(filename)

    
    # Interpolation model
    fig, ax = plt.subplots(1)
    ax.contour(x, y, nms3.model_polynomial.feval(x, y), levels)
    plt.scatter(vt[0, :], vt[1, :])
    filename = './plot_interpolation_model.png'
    plt.savefig(filename)
    # ball.generate_vectors_with_uniform_angles_svd(5, 2)
    
    center = nms3.sample_set.ball.center
    radius = nms3.sample_set.ball.rad
    intx = x - center[0]
    inty = y - center[1]
    dist = np.sqrt(intx**2 + inty**2)
    intindices = dist <= radius
    
    
    # Original function with interpolation model
    fig, ax = plt.subplots(1)
    
    func = myfunc([x, y])
    func[intindices] = nms3.model_polynomial.feval(x, y)[intindices]
    
    circle2 = plt.Circle(center, radius, color='blue')
    ax.add_patch(circle2)
    ax.contour(x, y, func, levels)
    plt.scatter(vt[0, :], vt[1, :])
    filename = './plot_function_w_interpolation.png'
    plt.savefig(filename)
    # ball.generate_vectors_with_uniform_angles_svd(5, 2)
    