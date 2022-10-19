import sys
import csv

sys.path.extend(['../utils'])
from math_functions import logsumexp, log, abs, ceil, sum, norm, exp, mod, dot, random, sqrt, std, append, \
    multivariate_normal, vectorise
from math_functions import gamma as gammagen
from plotting_functions import plt, plot_evolution, plot_box_plot
import numpy as np
from tqdm import tqdm


def eval_prior(x, alpha0=1., beta0=1.):
    """ Evaluate prior over samples """
    # return multivariate_normal.pdf(np.squeeze(x), mean=np.zeros(x.shape[0]), cov=np.eye(x.shape[0]))  # Choice of prior?
    pass


def sample_prior(dims=1):
    return vectorise(random.randn(dims))


def eval_proposal(new, old, n):
    """ Evaluate  proposal q(new|old) """
    """ Symmetric proposal: q(new|old) = N(old, I)"""
    # return multivariate_normal.pdf(np.squeeze(new), mean=np.squeeze(old), cov=np.eye(old.shape[0]) * n ** (-0.5))
    pass


def sample_proposal(old, n):
    """ Sample from proposal q(.|old)"""
    return old + vectorise(np.random.randn(old.shape[0])) * n ** (-0.5)


def ll_grad(y, x, kappa, sample):
    """ Evaluate gradient of log likelihood at x"""
    """ Model is Gamma Regression Model """
    return kappa * (y * np.squeeze(exp(-dot(sample.T, x))) - 1.) * x


def ll_hessian(y, x, kappa, sample):
    """ Evaluate second derivative of log likelihood at x"""
    return -kappa * y * np.squeeze(exp(-dot(sample.T, x))) * x @ x.T


def proxy(y, x, new, old, kappa, theta_s):
    """ Evaluate proxy at theta' = new, theta = old for x"""
    return dot(ll_grad(y, x, kappa, theta_s).T, (new - old)) + .5 * (new - old).T @ ll_hessian(y, x, kappa, theta_s) @ (
            old + new - 2. * theta_s)



def thirdDerivativeBound(ym, xm, sampled_xs, t1_prev, t2_prev, theta, kappa, theta_d, theta_s):
    t1 = min(t1_prev, min([dot(x.T, theta_d) for x in sampled_xs]))
    t2 = min(t2_prev, min([dot(x.T, theta) for x in sampled_xs]))
    t3 = max([np.sum(abs(theta_d_i - theta_s_i)) for theta_d_i, theta_s_i in zip(theta_d, theta_s)]) ** 3
    t4 = max([np.sum(abs(theta_i - theta_s_i)) for theta_i, theta_s_i in zip(theta, theta_s)]) ** 3
    return (kappa * ym * (xm**3) * (exp(-t1) * t3 + exp(-t2) * t4)) / 6., t1, t2

def concentration(batch_ll, ym, xm, sampled_xs, t1_prev, t2_prev, kappa, theta_s, theta, theta_d, delta, t):
    """ Evaluate concentration inequality """
    sigma = std(batch_ll)
    # C, t1, t2 = thirdDerivativeBound(ym, xm, sampled_xs, t1_prev, t2_prev, theta, kappa, theta_d, theta_s)
    C = max(abs(batch_ll))
    const = log(3. / delta) / t
    return sigma * sqrt(2. * const) + 6. * C * const, 0, 0


def log_likelihood(y, x, kappa, theta_s):
    """ Evaluate likelihood at x given parameters param """
    return -kappa * y * np.squeeze(exp(-dot(theta_s.T, x))) - kappa * np.squeeze(dot(theta_s.T, x))


def findDerivatives(ys, xs, kappa, theta_star):
    """ Return mean gradient and mean Hessian for entire dataset BEFORE starting MCMC """
    grads = kappa * np.squeeze(
        [(y * np.squeeze(exp(-dot(x.T, theta_star))) - 1.) * x for y, x in zip(ys, xs)])  # Find derivative
    Hessians = -kappa * np.squeeze(
        [(y * np.squeeze(exp(-dot(x.T, theta_star)))) * x @ x.T for y, x in zip(ys, xs)])  # Find second derivatives
    return vectorise(np.sum(grads, axis=0) / xs.shape[0]), np.sum(Hessians, axis=0) / xs.shape[0]


def sample_data(ys, xs, batch_size):
    """Sample |size| datapoints from data with replacement """
    # random.seed(1)  # Seed for reproducibility of re-sampled data
    indxs = random.randint(0, ys.shape[0], size=batch_size)
    sampled_y, sampled_x = ys[indxs], xs[indxs]
    ys = vectorise(np.delete(ys, indxs))  # Remove data from indices
    xs = np.delete(xs, indxs, axis=0)  # Remove data from indices
    return sampled_y, sampled_x, ys, xs


def initialiseMCMC(prev_sample, n):
    t = 0
    t_look = 0  # Number of batches used to approximate acceptance probability
    batch_size = 1  # Batch size
    new_sample = sample_proposal(prev_sample, n)
    u = random.uniform(low=0., high=1.)
    threshold = log(u) / n  # Flat prior, symmetric proposal
    av_log_likelihood = 0.
    return prev_sample, new_sample, threshold, batch_size, t, t_look, av_log_likelihood


def confidenceSampler(prev_theta, theta_s, gamma, ys, xs, ym, xm, data_size, mean_grad, mean_hess, delta, kappa, Recomputed):
    # TODO: Data will be large -> unfeasible to pass as array
    theta, theta_d, threshold, b, t, t_look, av_ll = initialiseMCMC(prev_sample=prev_theta, n=data_size)
    av_proxy = np.squeeze(mean_grad.T @ (theta_d - theta) + .5 * (theta_d - theta).T @ mean_hess @ (
            theta_d + theta - 2. * theta_s))
    isDone = False
    n = data_size
    batch_lls = np.array([])
    t1_prev = np.inf
    t2_prev = np.inf
    while (not isDone) and b < n:
        """ Sample with replacement"""
        # print(b, n, ys.shape[0])
        if Recomputed:
            b = n
        sampled_ys, sampled_xs, ys, xs = sample_data(ys, xs, b)
        sampled_batch_lls = np.array(
            [log_likelihood(y, x, kappa, theta_d) - log_likelihood(y, x, kappa, theta) - proxy(y,
                                                                                               x,
                                                                                               theta_d,
                                                                                               theta,
                                                                                               kappa, theta_s)
             for y, x in zip(sampled_ys, sampled_xs)])
        av_ll = (t * av_ll + sum(sampled_batch_lls)) / b
        batch_lls = append(batch_lls, sampled_batch_lls)
        t = b
        t_look += 1
        b = min(n, int(ceil(gamma * t)))
        c, t1_prev, t2_prev = concentration(batch_lls, ym, xm, sampled_xs, t1_prev, t2_prev, kappa, theta_s, theta, theta_d, delta / 2. / ((t_look) ** 2), t)
        if abs(av_ll + av_proxy - threshold) > c:
            isDone = True

    if (av_ll + av_proxy) > threshold:
        theta = theta_d
    return theta, b


def load_data():
    with open('GammaRegression.csv', 'r') as f_object:
        writer_object = csv.reader(f_object)
        i = 0
        xs = np.array(np.empty((1, 9, 1)))
        for row in writer_object:
            if i == 0:
                ys = np.array([float(x) for x in row])
                i += 1
            else:
                x = vectorise(np.array([float(ele) for ele in row]))
                xs = np.vstack([xs, x.reshape(1, x.shape[0], x.shape[1])])
        f_object.close()
    return ys.reshape(ys.shape[0], 1), xs, xs.shape[0]

def MH_chain(ys, xs, samples, kappa, N_iter=10000, gamma=2., drop_rate=10, delta=0.1):
    data_size = ys.shape[0]
    numEvals = np.array([0.])
    prop_evals = np.array([])
    online_posterior_means = np.array([samples[-1]])
    # Compute bounds on entire dataset
    ym = max(abs(ys))
    xm = max([max(x) for x in xs])
    for i in tqdm(range(N_iter)):
        if mod(i, drop_rate) == 0:
            theta_star = samples[-1]
            mean_gradient, mean_Hessian = findDerivatives(ys, xs, kappa=kappa, theta_star=theta_star)

        accepted_sample, b = confidenceSampler(samples[-1], theta_star, gamma, ys, xs, ym, xm, data_size, mean_gradient,
                                               mean_Hessian, delta, kappa, mod(i, drop_rate) == 0)
        print(i, b-data_size)
        if numEvals[-1] == 2 * data_size:
            numEvals = append(numEvals, b)
        else:
            numEvals = append(numEvals, 2 * b)

        prop_evals = append(prop_evals, numEvals[-1] / data_size)
        samples = np.vstack([samples, accepted_sample.reshape((1, accepted_sample.shape[0], accepted_sample.shape[1]))])
        online_posterior_means = np.vstack(
            [online_posterior_means, np.mean(samples, axis=0).reshape((1, samples[0].shape[0], samples[0].shape[1]))])
    return online_posterior_means, numEvals, prop_evals


def main(kappa=1.):
    prop_evals = []
    evols = []
    data_size = 10000
    true_theta = sample_prior(9)
    xs = np.array([np.atleast_2d(norm.rvs(size=9)).T for _ in range(data_size)])
    samples = np.array([90*vectorise(random.randn(9))])
    ys = np.atleast_2d(
        np.squeeze([gammagen.rvs(a=kappa, scale=kappa * np.squeeze(exp(dot(true_theta.T, x))), size=1) for x in xs])).T
    for i in range(1):
        opm, evals, pe = MH_chain(ys=ys, xs=xs, samples=samples, kappa=kappa)
        prop_evals.append(pe)
        evols.append(([np.squeeze(mean[0]) for mean in opm], np.cumsum(evals)))
    plot_evolution(evols)
    plot_box_plot(prop_evals)
    plt.show()


if __name__ == "__main__":
    main()
