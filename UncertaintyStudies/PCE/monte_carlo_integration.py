from problem_formulation import joint
from matplotlib import pyplot

sobol_samples = joint.sample(10000, rule="sobol")
antithetic_samples = joint.sample(10000, antithetic=True, seed=1234)
halton_samples = joint.sample(10000, rule="halton")

pyplot.figure(figsize=[16, 4])

pyplot.subplot(131)
pyplot.scatter(*sobol_samples[:, :1000])
pyplot.title("sobol")

pyplot.subplot(132)
pyplot.scatter(*antithetic_samples[:, :1000])
pyplot.title("antithetic variates")

pyplot.subplot(133)
pyplot.scatter(*halton_samples[:, :1000])
pyplot.title("halton")
pyplot.savefig("./PCE/mc_samples.png")
pyplot.close()

from problem_formulation import model_solver, coordinates
import numpy

sobol_evals = numpy.array([
    model_solver(sample) for sample in sobol_samples.T])

antithetic_evals = numpy.array([
    model_solver(sample) for sample in antithetic_samples.T])

halton_evals = numpy.array([
    model_solver(sample) for sample in halton_samples.T])

pyplot.figure(figsize=[16, 4])
pyplot.subplot(131)
pyplot.plot(coordinates, sobol_evals[:100].T, alpha=0.3)
pyplot.title("sobol")

pyplot.subplot(132)
pyplot.plot(coordinates, antithetic_evals[:100].T, alpha=0.3)
pyplot.title("antithetic variate")

pyplot.subplot(133)
pyplot.plot(coordinates, halton_evals[:100].T, alpha=0.3)
pyplot.title("halton")

pyplot.savefig("./PCE/mc_evaluations.png")
pyplot.close()


from problem_formulation import error_in_mean, indices, eps_mean

eps_sobol_mean = [error_in_mean(
    numpy.mean(sobol_evals[:idx], 0)) for idx in indices]

eps_antithetic_mean = [error_in_mean(
    numpy.mean(antithetic_evals[:idx], 0)) for idx in indices]

eps_halton_mean = [error_in_mean(
    numpy.mean(halton_evals[:idx], 0)) for idx in indices]


pyplot.figure()
pyplot.semilogy(indices, eps_mean, "r", label="random")
pyplot.semilogy(indices, eps_sobol_mean, "-", label="sobol")
pyplot.semilogy(indices, eps_antithetic_mean, ":", label="antithetic")
pyplot.semilogy(indices, eps_halton_mean, "--", label="halton")

pyplot.legend()
pyplot.savefig("./PCE/mc_mean.png")
pyplot.close()

from problem_formulation import error_in_variance, eps_variance

eps_halton_variance = [error_in_variance(
    numpy.var(halton_evals[:idx], 0)) for idx in indices]

eps_sobol_variance = [error_in_variance(
    numpy.var(sobol_evals[:idx], 0)) for idx in indices]

eps_antithetic_variance = [error_in_variance(
    numpy.var(antithetic_evals[:idx], 0)) for idx in indices]

pyplot.figure()
pyplot.semilogy(indices, eps_variance, "r", label="random")
pyplot.semilogy(indices, eps_sobol_variance, "-", label="sobol")
pyplot.semilogy(indices, eps_antithetic_variance, ":", label="antithetic")
pyplot.semilogy(indices, eps_halton_variance, "--", label="halton")

pyplot.legend()
pyplot.savefig("./PCE/mc_var.png")
pyplot.close()