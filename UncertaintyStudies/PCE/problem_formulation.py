import numpy
from matplotlib import pyplot
import chaospy

coordinates = numpy.linspace(0, 10, 1000)


def model_solver(parameters):
    """
    Simple ordinary differential equation solver.

    Args:
        parameters (numpy.ndarray):
            Hyper-parameters defining the model initial
            conditions alpha and growth rate beta.
            Assumed to have ``len(parameters) == 2``.

    Returns:
        (numpy.ndarray):
            Solution to the equation.
            Same shape as ``coordinates``.
    """
    alpha, beta = parameters
    return alpha*numpy.e**-(coordinates*beta)





pyplot.plot(coordinates, model_solver([1.5, 0.1]))
pyplot.plot(coordinates, model_solver([1.7, 0.2]))
pyplot.plot(coordinates, model_solver([1.7, 0.1]))
pyplot.savefig("./PCE/pf_simple_plots.png")
pyplot.close()

alpha = chaospy.Normal(1.5, 0.2)
beta = chaospy.Uniform(0.1, 0.2)
joint = chaospy.J(alpha, beta)


grid = numpy.mgrid[1:2:100j, 0.1:0.2:100j]
contour = pyplot.contourf(grid[0], grid[1], joint.pdf(grid), 50)
pyplot.scatter(*joint.sample(50, seed=1234))
pyplot.savefig("./PCE/pf_joint_dist.png")
pyplot.close()

parameter_samples = joint.sample(10000, seed=1234)
model_evaluations = numpy.array([model_solver(sample)
                                for sample in parameter_samples.T])
pyplot.plot(coordinates, model_evaluations[:1000].T, alpha=0.03)
pyplot.savefig("./PCE/pf_many_evaluations.png")
pyplot.close()

_t = coordinates[1:]

true_mean = numpy.hstack([
    1.5, 15*(numpy.e**(-0.1*_t)-numpy.e**(-0.2*_t))/_t])
true_variance = numpy.hstack([
    2.29, 11.45*(numpy.e**(-0.2*_t)-numpy.e**(-0.4*_t))/_t])-true_mean**2

std = numpy.sqrt(true_variance)
pyplot.fill_between(coordinates, true_mean-2*std, true_mean+2*std, alpha=0.4)
pyplot.plot(coordinates, true_mean)
pyplot.savefig("./PCE/pf_simple_band.png")
pyplot.close()

def error_in_mean(predicted_mean, true_mean=true_mean):
    """
    How close the estimated mean is the to the true mean.

    Args:
        predicted_mean (numpy.ndarray):
            The estimated mean.
        true_mean (numpy.ndarray):
            The reference mean value. Must be same shape as
            ``prediction_mean``.

    Returns:
        (float):
            The mean absolute distance between predicted
            and true values.
    """
    return numpy.mean(numpy.abs(predicted_mean-true_mean))

def error_in_variance(predicted_variance,
                      true_variance=true_variance):
    """
    How close the estimated variance is the to the true variance.

    Args:
        predicted_variance (numpy.ndarray):
            The estimated variance.
        true_variance (numpy.ndarray):
            The reference variance value.
            Must be same shape as ``predicted_variance``.

    Returns:
        (float):
            The mean absolute distance between
            predicted and true values.
    """
    return numpy.mean(numpy.abs(predicted_variance-true_variance))

indices = numpy.arange(100, 10001, 100, dtype=int)
eps_mean = [error_in_mean(numpy.mean(model_evaluations[:idx], 0))
            for idx in indices]
eps_variance = [error_in_variance(numpy.var(model_evaluations[:idx], 0))
                for idx in indices]

pyplot.semilogy(indices, eps_mean, "-", label="mean")
pyplot.semilogy(indices, eps_variance, "--", label="variance")
pyplot.legend()
pyplot.savefig("./PCE/pf_mean_var.png")