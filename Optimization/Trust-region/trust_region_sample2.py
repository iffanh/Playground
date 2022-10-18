import json
from lagrange_polynomial import LagrangePolynomials
from trust_region import TrustRegion, SubProblem
import matplotlib.pyplot as plt

import numpy as np

# def myfunc(x:np.ndarray) -> np.ndarray:
#     return x[0] + x[1] + 1*x[0]**2 + - 0*x[1]**3 + 3*x[1]**4

def myfunc(x:np.ndarray) -> np.ndarray:
    return x[0] - x[1] + 2*np.sin(x[0]) + 5*np.cos(x[1])

if __name__ == '__main__':

    print(f"==============================================")
    # Define scalars

    # dataset = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0], [0.0, 2.0]]).T
    dataset = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0], [0.0, -1.0]]).T
    # dataset = np.array([[0.524, 0.0006], [0.032, 0.323], [0.187, 0.890], [0.5, 0.5], [0.982, 0.368], [0.774, 0.918]]).T
    results = []
    for i in range(dataset.shape[1]):
        x = dataset[:, i]
        results.append(myfunc(x))
    results = np.array(results)
    
    print(f"Initial dataset: \n {dataset}")
    print(f"Initial function evaluation: \n {results}")    
    
    x, y = np.meshgrid(np.linspace(-4, 4, 100),
                    np.linspace(-4, 4, 100))
    
    levels = list(range(-40, 40, 2))
    
    tr = TrustRegion(dataset, results, myfunc)
    
    tr.run(max_radius=2.0, max_iter=20)
    
    old_center = [0,0]
    old_radius = 0
    
    for i, m in enumerate(tr.list_of_models):
        center = m.sample_set.ball.center
        radius = m.sample_set.ball.rad
        intx = x - center[0]
        inty = y - center[1]
        dist = np.sqrt(intx**2 + inty**2)
        intindices = dist <= radius
        
        fig, ax = plt.subplots(1)
        ax.set_title(tr.list_of_status[i])
        func = myfunc([x, y])
        func[intindices] = m.model_polynomial.feval(x, y)[intindices]
        circle1 = plt.Circle(center, radius, color='black', fill=False)
        circle2 = plt.Circle(old_center, old_radius, color='red', fill=False)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.contour(x, y, func, levels)
        plt.scatter(m.y[0, :], m.y[1, :], label=f'iteration_{i}')
        
        plt.legend()
        plt.savefig(f"./plots/TR_plots_{i}.png")
        
        old_center = center*1
        old_radius = radius*1