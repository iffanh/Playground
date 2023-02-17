import chaospy
from problem_formulation import joint

gauss_quads = [
    chaospy.generate_quadrature(order, joint, rule="gaussian")
    for order in range(1, 8)
]

sparse_quads = [
    chaospy.generate_quadrature(
        order, joint, rule=["genz_keister_24", "clenshaw_curtis"], sparse=True)
    for order in range(1, 5)
]

from matplotlib import pyplot

pyplot.figure(figsize=[12, 4])

nodes, weights = gauss_quads[5]

pyplot.subplot(121)
pyplot.title("Gaussian")
pyplot.scatter(*nodes, s=weights*2e3)

nodes, weights = sparse_quads[3]
idx = weights > 0

pyplot.subplot(122)
pyplot.title("sparse-grid")
pyplot.scatter(*nodes[:, idx], s=weights[idx]*2e3)
pyplot.scatter(*nodes[:, ~idx], s=-weights[~idx]*2e3, color="grey")

pyplot.savefig("./PCE/ps_quadratures.png")
pyplot.close()

from problem_formulation import model_solver, coordinates
import numpy

gauss_evals = [
    numpy.array([model_solver(node) for node in nodes.T])
    for nodes, weights in gauss_quads
]

sparse_evals = [
    numpy.array([model_solver(node) for node in nodes.T])
    for nodes, weights in sparse_quads
]
pyplot.figure(figsize=[12, 4])
pyplot.subplot(121)
pyplot.plot(coordinates, gauss_evals[6].T, alpha=0.3)
pyplot.title("Gaussian")

pyplot.subplot(122)
pyplot.plot(coordinates, sparse_evals[3].T, alpha=0.3)
pyplot.title("sparse-grid")

pyplot.savefig("./PCE/ps_quadratures_evals.png")
pyplot.close()

expansions = [chaospy.generate_expansion(order, joint)
              for order in range(1, 10)]
expansions[0].round(10)

gauss_model_approx = [
    chaospy.fit_quadrature(expansion, nodes, weights, evals)
    for expansion, (nodes, weights), evals in zip(expansions, gauss_quads, gauss_evals)
]

sparse_model_approx = [
    chaospy.fit_quadrature(expansion, nodes, weights, evals)
    for expansion, (nodes, weights), evals in zip(expansions, sparse_quads, sparse_evals)
]

model_approx = gauss_model_approx[4]
nodes, _ = gauss_quads[4]
evals = model_approx(*nodes)

pyplot.figure(figsize=[12, 4])
pyplot.subplot(121)
pyplot.plot(coordinates, evals, alpha=0.3)
pyplot.title("Gaussian")

model_approx = sparse_model_approx[1]
nodes, _ = sparse_quads[1]
evals = model_approx(*nodes)

pyplot.subplot(122)
pyplot.plot(coordinates, evals, alpha=0.3)
pyplot.title("sparse-grid")

pyplot.savefig("./PCE/ps_polychaos_evals.png")
pyplot.close()

expected = chaospy.E(gauss_model_approx[-2], joint)
std = chaospy.Std(gauss_model_approx[-2], joint)

expected[:4].round(4), std[:4].round(4)

pyplot.figure(figsize=[6, 4])

pyplot.xlabel("coordinates")
pyplot.ylabel("model approximation")
pyplot.fill_between(
    coordinates, expected-2*std, expected+2*std, alpha=0.3)
pyplot.plot(coordinates, expected)

pyplot.savefig("./PCE/ps_eval_band.png")
pyplot.close()

from problem_formulation import error_in_mean, error_in_variance

error_in_mean(expected), error_in_variance(std**2)

gauss_sizes = [len(weights) for _, weights in gauss_quads]
eps_gauss_mean = [
    error_in_mean(chaospy.E(model, joint))
    for model in gauss_model_approx
]
eps_gauss_var = [
    error_in_variance(chaospy.Var(model, joint))
    for model in gauss_model_approx
]

sparse_sizes = [len(weights) for _, weights in sparse_quads]
eps_sparse_mean = [
    error_in_mean(chaospy.E(model, joint))
    for model in sparse_model_approx
]
eps_sparse_var = [
    error_in_variance(chaospy.Var(model, joint))
    for model in sparse_model_approx
]

pyplot.figure(figsize=[12, 4])

pyplot.subplot(121)
pyplot.title("Error in mean")
pyplot.loglog(gauss_sizes, eps_gauss_mean, "o-", label="Gaussian")
pyplot.loglog(sparse_sizes, eps_sparse_mean, "o--", label="sparse")
pyplot.legend()

pyplot.subplot(122)
pyplot.title("Error in variance")
pyplot.loglog(gauss_sizes, eps_gauss_var, "o-", label="Gaussian")
pyplot.loglog(sparse_sizes, eps_sparse_var, "o--", label="sparse")

pyplot.savefig("./PCE/ps_error_mean_var.png")
pyplot.close()