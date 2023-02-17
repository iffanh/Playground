import numpy as np
import casadi as ca
import chaospy

import lagrange_polynomial as lp

# def f(x:np.ndarray, a:float, b:float): # Assume long running process
#     return np.abs(a*x[0] + b*np.exp(-x[1]))

def f(x:np.ndarray, a, b) -> np.ndarray:
    return a*(x[0]**2)*(1 + 0.75*np.cos(70*x[0])/12) + np.cos(100*x[0])**2/24 + b*(x[1]**2)*(1 + 0.75*np.cos(70*x[1])/12) + np.cos(100*x[1])**2/24 + 4*x[0]*x[1]

samples = np.array([[0.0, 0.0], 
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0]]).T 


## Specify uncertainty, quadratures, and PC expansion
alpha = chaospy.Normal(10, 2.0)
beta = chaospy.Uniform(1.5, 2.5)
joint = chaospy.J(alpha, beta)
gauss_quads = chaospy.generate_quadrature(3, joint, rule="gaussian")
nodes, weights = gauss_quads
expansion = chaospy.generate_expansion(2, joint)


# Build Lagrange polynomials for decision variables
# lpe = chaospy.expansion.lagrange(samples)

input_symbols = ca.SX.sym('x', samples.shape[0])
lag_polys = lp.LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
lag_polys.initialize(samples)
lpe = lag_polys.lagrange_polynomials


## Build PCE for uncertainty

model = ca.SX(0)
model_db = dict()
for i, x in enumerate(samples.T):
    model_db[f"sample_{i}"] = dict()
    model_db[f"sample_{i}"]["point"] = samples[:, i]
    gauss_evals = np.array([f(x, node[0], node[1]) for node in nodes.T])
    model_db[f"sample_{i}"]["gauss_evals"] = gauss_evals
    gauss_model_approx = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals)
    model_db[f"sample_{i}"]["gauss_model_approx"] = gauss_model_approx
    gauss_model_approx_evals = np.array([gauss_model_approx(node[0], node[1]) for node in nodes.T])
    model_db[f"sample_{i}"]["gauss_model_approx_evals"] = gauss_model_approx_evals
    expected = chaospy.E(gauss_model_approx, joint)
    std = chaospy.Std(gauss_model_approx, joint)
    model_db[f"sample_{i}"]["expected"] = expected
    model_db[f"sample_{i}"]["std"] = std
    model_db[f"sample_{i}"]["lag_poly"] = lpe[i]

    # print(f"gauss_evals*lp[i] = {gauss_evals*lp[i]}")
    # model = model + gauss_evals*lpe[i].symbol
    model = model + gauss_model_approx_evals*lpe[i].symbol

model_approx = ca.Function('approx', [input_symbols], [model])


## Build surface, one for each sample model.
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
fsize = 10
tsize = 18
tdir = 'in'
major = 5.0
minor = 3.0
style = 'default'
plt.style.use(style)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
# make grid
X, Y = np.meshgrid(np.linspace(-3, 3, 81),
                    np.linspace(-3, 3, 101))
gridsize = X.shape[0]*X.shape[1] 
surfaces = model_approx.map(gridsize)(ca.horzcat(X.reshape(gridsize), Y.reshape(gridsize)).T)

Xflat = X.flatten()
Yflat = Y.flatten()

surf = []
for k in range(gridsize):
    surf.append(model_approx([Xflat[k], Yflat[k]]).full()[:,0])
surf = np.array(surf) #.reshape((X.shape[0], X.shape[1]))


levels = [10**i for i in np.arange(-1.0,2.0,0.1)]

for j in range(model.shape[0]):
    # surface = surfaces[j, :].reshape((X.shape[0], X.shape[1]))
    surface = surf[:,j].reshape((X.shape[0], X.shape[1]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title(f"$\\alpha$ = {nodes[0, j]}, \n $\\beta = {nodes[1, j]}$")
    ax[0].set_xlabel(f"$x_1$")
    ax[0].set_ylabel(f"$x_2$")
    CS = ax[0].contour(X, Y, surface, levels, norm = LogNorm(), cmap=cm.PuBu_r)
    ax[0].clabel(CS, fontsize=9, inline=True)
    ax[0].scatter(samples[0, :], samples[1, :], color='black')
    # plt.scatter(x[0], y[0], color='red', label=f'Best point')
    
    act_surface = np.array([f([xv, yv], nodes[0, j], nodes[1, j]) for xv, yv in zip(X.reshape(gridsize), Y.reshape(gridsize))]).reshape((X.shape[0], X.shape[1]))
    ax[1].set_title(f"$\\alpha$ = {nodes[0, j]}, \n $\\beta = {nodes[1, j]}$")
    ax[1].set_xlabel(f"$x_1$")
    ax[1].set_ylabel(f"$x_2$")
    CS = ax[1].contour(X, Y, act_surface, levels, norm = LogNorm(), cmap=cm.PuBu_r)
    ax[1].clabel(CS, fontsize=9, inline=True)
    ax[1].scatter(samples[0, :], samples[1, :], color='black')
    plt.savefig(f'./PCE/surfaces/surface_{j}')
    plt.close()