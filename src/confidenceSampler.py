import sys

sys.path.extend(['../utils'])
from math_functions import logsumexp, log, abs, ceil, max, sum
import numpy as np


def prior(x):
    """ Evaluate prior over samples """
    pass


def proposal(new, old):
    """ Evaluate  proposal q(new|old) """
    pass


def sample_proposal(old):
    """ Sample from proposal q(.|old)"""
    pass


def ll_grad(x):
    """ Evaluate gradient of log likelihood at x"""
    pass


def ll_hessian(x):
    """ Evaluate second derivative of log likelihood at x"""
    pass


def proxy(x, new, old):
    """ Evaluate proxy at theta' = new, theta = old for x"""
    return ll_grad(x) * (old - new) + ll_hessian(x)*(old - new) ** 2 / 2.


def sample_data(data, size):
    """Sample |size| datapoints from data with replacement """
    pass


def concentration(delta):
    """ Evaluate concentration inequality """
    pass


def likelihood(x, param):
    """ Evaluate likelihood at x given parameters param """
    pass


def initialiseMCMC(prev_sample, n):
    t = 0
    t_look = 0  # Number of batches used to approximate acceptance probability
    batch_size = 1  # Batch size
    sample = prev_sample
    new_sample = sample_proposal(sample)
    u = np.random.uniform(low=0., high=1.)
    threshold = (log(u) + log(prior(sample)) - log(prior(new_sample)) + log(proposal(new_sample, sample)) - log(
        proposal(sample, new_sample))) / n
    av_log_likelihood = 0
    return sample, new_sample, threshold, batch_size, t, t_look, av_log_likelihood


def confidenceSampler(prev_theta, gamma, data, data_size, mean_grad, mean_hess, delta):
    # TODO: Data will be large -> unfeasible to pass as array
    theta, theta_d, threshold, b, t, t_look, av_ll = initialiseMCMC(prev_sample = prev_theta, n=data_size)
    av_proxy = mean_grad * (theta - theta_d) + 0.5 * (
            theta - theta_d) ** 2 * mean_hess  # TODO: Check dimensions (assume 1d for now)
    isDone = False
    n = data_size
    while not isDone:
        """ Sample with replacement"""
        sampled_data = sample_data(data, b)
        batch_ll = sum(
            [log(likelihood(x, theta_d)) - log(likelihood(x, theta)) - proxy(x, theta_d, theta) for x in sampled_data])
        av_ll = (t * av_ll + batch_ll) / b
        t += b
        c = concentration(delta) # TODO: Here assuming constant delta
        t_look += 1
        b += max(n, ceil(gamma * t))
        if abs(av_ll + av_proxy - threshold) >= c or t == n:  # TODO: Check precision issues
            isDone = True
        pass
    if av_ll > (threshold - av_proxy):
        theta = theta_d
    return theta


def load_data():
    """ Return dataset, and its size """
    pass


def preprocessing(data):
    """ Return mean gradient and mean Hessian for entire dataset BEFORE starting MCMC """
    pass


def main(N_iter=10000, gamma=0.5):
    samples = np.array([])
    data, n = load_data()
    mean_gradient, mean_Hessian = preprocessing(data)
    delta = 0.9
    for i in range(N_iter):
        samples = np.append(samples, confidenceSampler(samples[-1],gamma, n, mean_gradient, mean_Hessian, delta))


if __name__ == "__main__":
    main(N_iter=1, gamma=1.0)
