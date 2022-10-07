import json
from lagrange_polynomial import LagrangePolynomials
from trust_region import TrustRegion, SubProblem
import matplotlib.pyplot as plt

import numpy as np

def myfunc(x:np.ndarray) -> np.ndarray:
    return x[0] + x[1] + 2*x[0]**2 + 3*x[1]**3

if __name__ == '__main__':

    print(f"==============================================")
    # Define scalars

    # dataset = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0], [0.0, 2.0]]).T
    # dataset = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0], [0.0, -1.0]]).T
    dataset = np.array([[0.524, 0.0006], [0.032, 0.323], [0.187, 0.890], [0.5, 0.5], [0.982, 0.368], [0.774, 0.918]]).T
    results = []
    for i in range(dataset.shape[1]):
        x = dataset[:, i]
        results.append(myfunc(x))
    results = np.array(results)
    
    print(f"Initial dataset: \n {dataset}")
    print(f"Initial function evaluation: \n {results}")    
    
    x, y = np.meshgrid(np.linspace(-4, 4, 100),
                    np.linspace(-4, 4, 100))
    # levels = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    levels = list(range(-20, 20, 2))
    iteration_data = dict()
    max_radius = 1.0
    eta = 0.2
    iteration_data["Initial"] = dict()
    
    for k in range(5):
        
        tr = TrustRegion(dataset, results, myfunc)
        
        if k == 0:
            new_radius = tr.polynomial.sample_set.ball.rad
        
        tr.solve_subproblem(rad=new_radius)
        
        path = tr.sp.path
        opt_point = tr.center + path
        step_size = np.linalg.norm(path)
        
        m1 = tr.model_evaluation(tr.center)
        m2 = tr.model_evaluation(opt_point)
        f1 = myfunc(tr.center)
        f2 = myfunc(opt_point)
        rho = tr.calculate_rho_ratio(f1, f2, m1, m2)
        
        new_radius = tr.get_new_radius(rho=rho, step_size=step_size, max_radius=max_radius)
        new_point = tr.get_new_point(rho=rho, eta=eta)
        
        iteration_data[f"{k}"] = dict()
        iteration_data[f"{k}"]["ball_center"] = tr.center
        iteration_data[f"{k}"]["ball_radius"] = tr.rad
        iteration_data[f"{k}"]["opt_point"] = opt_point
        iteration_data[f"{k}"]["path"] = path
        iteration_data[f"{k}"]["step_size"] = step_size
        iteration_data[f"{k}"]["m1"] = m1
        iteration_data[f"{k}"]["m2"] = m2
        iteration_data[f"{k}"]["f1"] = f1
        iteration_data[f"{k}"]["f2"] = f2
        iteration_data[f"{k}"]["rho"] = rho
        iteration_data[f"{k}"]["new_radius"] = new_radius
        iteration_data[f"{k}"]["new_point"] = new_point
        
        # What to do with the new point and function evaluation??
        # tr.dataset[:, 0] = new_point
        # tr.results[0] = f2
        # dataset = tr.dataset
        # results = tr.results
        
        # print(tr.results, tr.results.argsort(), type(tr.results))
        # sort_ind = np.argsort(list(tr.results), order='reverse')
        sort_ind = tr.results.argsort(axis=0)[::-1]
        dataset = tr.dataset[:, sort_ind]
        results = tr.results[sort_ind]
        
        worst_point = tr.dataset[:,0]
        dataset[:,0] = new_point
        results[0] = f2
        
        # dataset = np.concatenate([tr.dataset, np.array([new_point]).T], axis=1)
        # results = np.concatenate([tr.results, np.array([f2])])
        
        print(f"=============================")
        # print(f"Dataset after {k} iterations: \n {dataset}")
        # print(f"Function evaluation after {k} iterations: \n {results}")
        # print(f"Iteration information: {iteration_data[f'{k}']}")    
        
        
        center = tr.center
        radius = new_radius
        intx = x - center[0]
        inty = y - center[1]
        dist = np.sqrt(intx**2 + inty**2)
        intindices = dist <= radius
    
        # Original function with interpolation model
        fig, ax = plt.subplots(1)
        
        func = myfunc([x, y])
        func[intindices] = tr.polynomial.model_polynomial.feval(x, y)[intindices]
        
        circle2 = plt.Circle(center, radius, color='black', fill=False)
        ax.add_patch(circle2)
        ax.contour(x, y, func, levels)
        plt.scatter(dataset[0, :], dataset[1, :], color='black')
        plt.scatter(new_point[0], new_point[1], color='green')
        plt.scatter(worst_point[0], worst_point[1], color='red')
        plt.scatter(center[0], center[1], color='blue')
        filename = f'./plots/plot_function_w_interpolation_{k}.png'
        plt.savefig(filename)
    
# if __name__ == '__main__':
    
#     print(f"==============================================")
#     filename = "./data/test.json"
    
#     with open(filename, 'r') as f:
#         data = json.load(f)
        
#     print(data)