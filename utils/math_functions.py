import numpy as np
from numpy import broadcast_to, log, exp, abs, ceil, max, sum


def logsumexp(w, h, x, axis=0, retlog=False):
    c = np.max(w)
    broad_l = broadcast_to((w - c).flatten(), x.T.shape).T
    if retlog:
        return c + log((exp(broad_l) * h(x)).sum(axis=axis))
    return exp(c) * (exp(broad_l) * h(x)).sum(axis=axis)
