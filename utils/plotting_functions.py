import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_evolution(chain_ts, fig=None, ax=None):
    plt.style.use('ggplot')
    matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
    })
    if not (fig or ax):
        fig, ax = plt.subplots()
    for sample_dim, evals in chain_ts:
        ax.scatter(evals, sample_dim, linestyle='dashed', s=1)
    ax.set_xscale('log')
    ax.set_xlabel("Total Likelihood Evaluations")
    ax.set_ylabel("Online Posterior Mean")
    ax.set_title("Posterior Mean for First Dimension of $\\theta$")
    plt.grid(True, which="both", linestyle='--')


def plot_box_plot(data, fig=None, ax=None):
    if not (fig or ax):
        fig, ax = plt.subplots()
    indxs = np.arange(1, len(data) + 1)
    labels = ["Run " + str(i) for i in indxs]
    ax.boxplot(data, labels=labels)
    ax.set_yscale('log')
    ax.set_ylabel("Proportion of Likelihood Evaluations")
    ax.set_title("Box Plots Proportion of Likelihood Evaluations per MH iteration")
    ax.legend()
