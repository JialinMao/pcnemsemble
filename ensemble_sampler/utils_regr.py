# helper functions, all assignment specific

import numpy as np
import matplotlib.pyplot as plt


def generate_fake_data(A, l, s, t):
    """
    A function to generate fake data given parameters and time steps.

    Parameters:
    A, l, s : Parameters (s the nuisance parameter)
    t : time steps

    Returns:
    t, Y : Y[i] = sum_{j}(A[j] * exp(-l[j] * t[i])) + x_i
    """
    n = len(t)

    f_t = np.dot(np.exp(-np.outer(t, l)), A)
    xi = np.random.normal(loc=0.0, scale=s, size=n)
    Y = f_t + xi

    return {'t': t, 'Y': Y}


def log_likelihood(Y, t, p, log_prior=None, **kwargs):
    """
    Evaluate the likelihood of observation (t, Y) given parameters (A, l, s)
    If log_prior is not None, expecting prior_args (in form of dict) in kwargs
    """
    m = (len(p) - 1) / 2
    A = p[:m]
    l = p[m:-1]
    s = p[-1]
    x = Y - np.dot(np.exp(-np.outer(t, l)), A)
    log_prob = - x ** 2 / (2 * s ** 2) - np.log(max(abs(s), 1e-8))
    log_prob = np.sum(log_prob)
    if log_prior is not None:
        log_prob += log_prior(A, l, s, **kwargs.get('prior_args'))
    if kwargs.get('verbose', False):
        print log_prob
    return log_prob


def flat_prior(A, l, s, A_max=None, l_max=None, s_max=None):
    """
    Uniform prior for A in [-Amax, Amax], 0 for A outside the range. Similar for l, s
    Including the prior that std_dev should be positive
    """
    if np.all(abs(A) <= A_max) and np.all(abs(l) <= l_max) and np.all(0 < s <= s_max):
        return 0
    else:
        return -np.inf


def gaussian_prior(A, l, s, mean=None, cov=None):
    """
    Gaussian prior with large variance
    """
    x = np.hstack([A, l, s])
    mu = np.zeros(len(x)) if mean is None else mean
    c = 10 * np.identity(len(x)) if cov is None else cov
    inverse_c = np.linalg.inv(c)
    log_prob = np.dot(x - mu, np.dot(inverse_c, x - mu)) + np.log(np.linalg.det(c))
    return - log_prob / 2.0


def gaussian_proposal(p_0, var):
    """
    A multivariate gaussian proposal, mean p_0, covariance matrix var
    """
    return np.random.multivariate_normal(p_0, var)


def pcn_proposal(p_0, beta, cov):
    """
    A pCN proposal, used together with a gaussian prior
    """
    xi = beta * np.random.multivariate_normal(mean=np.zeros_like(p_0), cov=cov)
    return np.sqrt((1 - beta ** 2)) * p_0 + xi


def plot_hist(dim, history, p_0=None, Al=None, burn_in=0):
    """
    Plot histogram of history for chosen dimension(s). dim is an integer array.
    If Al is not None also plot the actual value (used to generate the fake data). (Same below)
    """
    n = len(dim)
    num_steps = len(history)
    nrows = int(np.ceil(n / 2.0))
    fig, axs = plt.subplots(nrows, 2, figsize=(15, 9))
    fig.suptitle("Sample history histogram, dim=%s, num_steps=%s" % (dim, num_steps))
    for i in range(len(dim)):
        idx = i if n == 2 else (i // 2, i % 2)
        axs[idx].hist(history[burn_in:, dim[i]], 100, histtype='step')
        if Al is not None:
            assert len(Al) == n
            axs[idx].axvline(Al[dim[i]], linestyle='--', color='r', linewidth=1)
            axs[idx].legend(['actual'], loc='lower right')
            if p_0 is not None:
                assert len(p_0) == n
                axs[idx].axvline(p_0[dim[i]], linestyle='--', color='g', linewidth=1)
                axs[idx].legend(['actual', 'start'], loc='lower right')
        axs[idx].set_title("Dim %s histogram" % int(i+1))
    plt.show()


def plot_trajectory(dim, history, Al=None, burn_in=0):
    """
    Plot the trajectory for given dimension(s). Just for fun.
    """
    N = len(history)
    n = len(dim)
    nrows = int(np.ceil(n / 2.0))
    fig, axs = plt.subplots(nrows, 2, figsize=(15, 9))
    fig.suptitle("Sample history trajectory, dim=%s" % dim)
    for i in range(len(dim)):
        idx = i if n == 2 else (i // 2, i % 2)
        traj, = axs[idx].plot(np.arange(N - burn_in), history[burn_in:, i])
        if Al is not None:
            assert len(Al) == n, 'dimension error, dim=%s, dim(Al)=%s' % (len(dim), len(Al))
            opt, = axs[idx].plot((0, N), (Al[i], Al[i]), '--', alpha=0.5)
            axs[idx].legend([opt, traj], ['actual', 'sample'], loc='lower right')
        axs[idx].set_title("Dim %s history" % int(i + 1))
    plt.show()


def plot_scattered(dim, history, Al=None):
    """
    Scattered plot for two chosen dimensions. Not sure whether this makes sense...
    dim should be list of pairs of integers [[a_1, b_1], [a_2, b_2], ...]
    """
    for i in dim:
        plt.figure()
        hist = plt.scatter(history[:, i[0]], history[:, i[1]], alpha=0.5)
        if Al is not None:
            assert len(Al) == len(dim), 'dimension error, dim=%s, dim(Al)=%s' % (len(dim), len(Al))
            opt, = plt.plot([Al[i[0]]], [Al[i[1]]], 'ro')
            plt.legend([opt, hist], ['actual', 'sample'], loc='lower right')
        plt.title('Dim%s_history' % int(i+1))

