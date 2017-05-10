import numpy as np
import matplotlib.pyplot as plt


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
        idx = i if n == 2 else (i // 2, i % 2)
        axs[idx].plot(np.arange(N - start_from), history[start_from:, i])
        axs[idx].set_title("Dim %s history" % int(i + 1))
    plt.show()
