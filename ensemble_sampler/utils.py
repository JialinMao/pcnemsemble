import numpy as np
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
import timeit
from emcee.autocorr import *

import ensemble_sampler as es

__all__ = ['plot_hist', 'plot_trajectory', 'plot_acf', 'run']


def plot_acf(chain, max_lag=1000, mean_first=False):
    if mean_first:
        acf = function(np.mean(chain, axis=0))
        title = 'Mean of walkers first'
    else:
        acf = np.zeros_like(chain)
        for i in range(acf.shape[0]):
            acf[i] = function(chain[i])
        acf = np.mean(acf, axis=0)
        title = 'Mean of acf'
    data = DataFrame(acf[:max_lag], columns=['dim_1','dim_2'])
    ax = data.plot()
    sns.set_style("whitegrid", {'axes.grid' : False})
    ax.set(xlabel='lag', ylabel='ACF', title=title)
    plt.show()


def plot_hist(dim, history, start_from=None, normalize=False, show=False):
    """
    Plot histogram of history for chosen dimension(s). dim is an integer array.
    """
    n = len(dim)
    num_steps = len(history)
    nrows = int(np.ceil(n / 2.0))
    fig, axs = plt.subplots(nrows, 2, figsize=(15, 9))
    fig.suptitle("Sample history histogram, dim=%s, num_steps=%s" % (dim, num_steps))
    for i in range(len(dim)):
        idx = i if n == 1 or n == 2 else (i // 2, i % 2)
        axs[idx].hist(history[start_from:, dim[i]], 100, histtype='step', normed=normalize)
        axs[idx].set_title("Dim %s histogram" % int(i+1))
    if show:
        plt.show()


def plot_trajectory(dim, history, start_from=0):
    """
    Plot the trajectory for given dimension(s). Just for fun.
    """
    N = len(history)
    n = len(dim)
    nrows = int(np.ceil(n / 2.0))
    fig, axs = plt.subplots(nrows, 2, figsize=(15, 9))
    fig.suptitle("Sample history trajectory, dim=%s" % dim)
    for i in range(len(dim)):
        idx = i if n == 1 or n == 2 else (i // 2, i % 2)
        axs[idx].plot(np.arange(N - start_from), history[start_from:, i])
        axs[idx].set_title("Dim %s history" % int(i + 1))
    plt.show()


def run(dim, sampler, batch_size=50, niters=1000, n=5, pre=0, nwalkers=100,
        title='', verbose=False, print_every=200, plot=False, save_dir=None,
        store=False, store_every=1):
    acc_r = 0.0
    for i in range(n):
        sampler.init()
        p0 = np.random.randn(dim*nwalkers).reshape([nwalkers, dim])
        if pre > 0:
            s = es.Sampler(dim=dim, t_dist=sampler.t_dist, proposal=es.WalkMove(s=None, scale=0.2), nwalkers=nwalkers)
            p0 = s.run_mcmc(pre, batch_size=batch_size, p0=p0, verbose=False).curr_pos
        start = timeit.default_timer()
        sampler.run_mcmc(niters-pre, batch_size=batch_size, p0=p0, verbose=verbose, print_every=print_every,
                         store=store, store_every=store_every, save_dir=save_dir, title=title)
        print 'finishes loop %d in %.2f seconds' % (i, float(timeit.default_timer() - start))
        acc_curr_iter = float(100*np.mean((sampler.acceptance / niters)))
        acc_r += acc_curr_iter
        try:
            auto_corr = sampler.auto_corr()
        except AutocorrError, err:
            auto_corr = err
        print 'auto-correlation time: %s' % auto_corr

    print 'avg_acc_r: %.2f%s' % (float(acc_r) / float(n), '%')

    if plot:
        img = sampler.history.plot_scatter(dim=[[0, 1]])
        sns.plt.title(title)
        fig = img.get_fig()
        fig.savefig(title)




