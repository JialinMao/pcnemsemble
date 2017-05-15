import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
from emcee.autocorr import *
import cPickle

import ensemble_sampler as es

__all__ = ['plot_hist', 'plot_trajectory', 'run']


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


def run(dim, sampler, batch_size=50, niters=1000, n=5, pre=0, nwalkers=100,
        title='', verbose=False, print_every=200, plot=False, save_dir=None, save_every=1):
    acc_r = 0.0
    for i in range(n):
        sampler.reset()
        p0 = np.random.randn(dim*nwalkers).reshape([nwalkers, dim])
        if pre > 0:
            s = es.Sampler(dim=dim, t_dist=sampler.t_dist, proposal=es.PCNWalkMove(s=None, scale=0.2), nwalkers=nwalkers)
            p0 = s.run_mcmc(pre, batch_size=batch_size, p0=p0, verbose=False).curr_pos
        start = timeit.default_timer()
        hist = sampler.run_mcmc(niters-pre, batch_size=batch_size, p0=p0, verbose=verbose, print_every=print_every)
        print 'finishes loop %d in %.2f seconds' % (i, float(timeit.default_timer() - start))
        acc_curr_iter = float(100*hist.acceptance_rate.mean())
        acc_r += acc_curr_iter
        try:
            auto_corr = hist.auto_corr()
        except AutocorrError, err:
            auto_corr = err
        finally:
            print 'auto-correlation time: %s' % auto_corr

        if save_dir is not None and i % save_every == 0:
            print 'writing to %s.pkl...' % title
            with open(save_dir+title+'_'+str(i)+'.pkl', 'wb') as f:
                cPickle.dump({'auto_corr': auto_corr,
                              'acceptance_rate': acc_curr_iter}, f)

    print 'avg_acc_r: %.2f%s' % (float(acc_r) / float(n), '%')

    if plot:
        img = hist.plot_scatter(dim=[[0, 1]])
        sns.plt.title(title)
        fig = img.get_fig()
        fig.savefig(title)


