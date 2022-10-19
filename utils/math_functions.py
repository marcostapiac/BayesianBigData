import numpy as np
from scipy.stats import norm, gamma
from scipy.stats import multivariate_normal, poisson, bernoulli
from numpy import broadcast_to, log, exp, abs, ceil, sum, mod, dot, random, sqrt, std, append


def logsumexp(w, h, x, axis=0, retlog=False):
    c = np.max(w)
    broad_l = broadcast_to((w - c).flatten(), x.T.shape).T
    if retlog:
        return c + log((exp(broad_l) * h(x)).sum(axis=axis))
    return exp(c) * (exp(broad_l) * h(x)).sum(axis=axis)


def vectorise(x):
    assert (len(x.shape) == 1)
    return np.atleast_2d(x).T

