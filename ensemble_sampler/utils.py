import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
from emcee.autocorr import *

from .distribution import *
from .sampler import *
from .proposal import *

__all__ = ['plot_hist', 'plot_trajectory', 'run_spde']


def plot_hist(dim, history, start_from=None):
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
        axs[idx].hist(history[start_from:, dim[i]], 100, histtype='step')
        axs[idx].set_title("Dim %s histogram" % int(i+1))
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


def run_spde(dim, proposal, batch_size=50, niters=1000, n=5, pre=0, nwalkers=100):
    t_dist = SPDE(N=dim)
    sampler = Sampler(dim=dim, t_dist=t_dist, proposal=proposal, nwalkers=nwalkers)

    acc_r = 0.0
    for i in range(n):
        sampler.reset()
        p0 = np.random.randn(dim*nwalkers).reshape([nwalkers, dim])
        if pre > 0:
            hist = sampler.run_mcmc(pre, batch_size=batch_size, p0=p0, verbose=False)
            p0 = hist.curr_pos
            sampler.reset()
        start = timeit.default_timer()
        hist = sampler.run_mcmc(niters, batch_size=batch_size, p0=p0, verbose=False)
        print 'finishes loop %d in %.2f seconds' % (i, float(timeit.default_timer() - start))
        acc_r += float(100*hist.acceptance_rate.mean())
        # print 'avg accept rate: %.2f%s' % (float(100*hist.acceptance_rate.mean()), '%')
        try:
            print 'auto-correlation time: %s' % hist.auto_corr()
        except AutocorrError:
            pass
    print 'avg_acc_r: %.2f%s' % (float(acc_r) / 5.0, '%')


def run_rosenbrock(proposal, batch_size=50, niters=1000, nwalkers=100, n=5, pre=0, title=''):
    sampler = Sampler(dim=2, t_dist=Rosenbrock(), proposal=proposal, nwalkers=nwalkers)

    acc_r = 0.0
    for i in range(n):
        sampler.reset()
        p0 = np.random.randn(2*nwalkers).reshape([nwalkers, 2])
        if pre > 0:
            s = Sampler(dim=2, t_dist=Rosenbrock(), proposal=PCNWalkMove(s=None, scale=0.2), nwalkers=nwalkers)
            p0 = s.run_mcmc(pre, batch_size=batch_size, p0=p0).curr_pos
        start = timeit.default_timer()
        hist = sampler.run_mcmc(niters-pre, batch_size=batch_size, p0=p0)
        print 'finishes loop %d in %.2f seconds' % (i, float(timeit.default_timer() - start))
        acc_r += float(100*hist.acceptance_rate.mean())
        # print 'avg accept rate: %.2f%s' % (float(100*hist.acceptance_rate.mean()), '%')
        try:
            print 'auto-correlation time: %s' % hist.auto_corr()
        except AutocorrError, err:
            pass
    hist.plot_scatter(dim=[[0, 1]])
    sns.plt.title(title)
    print 'avg_acc_r: %.2f%s' % (float(acc_r) / 5.0, '%')
