import matplotlib.pyplot as plt
import argparse
from ensemble_sampler.utils import *
from simple_sampler import *


def run(dim, nwalkers, beta, niter, rand, visualize=False, vis_every=200):
    p0 = rand.randn(nwalkers, dim)
    history = np.zeros([nwalkers, niter+1, dim])
    acceptances = np.zeros([nwalkers, niter])
    history[:, 0, :] = p0
    plt.ion()
    for h in sample(niter, p0, nwalkers, dim, beta):
        curr_walker, ensemble, proposal, accept, ens_icov, ens_mean, i, k = h
        history[:, i+1, :] = np.copy(history[:, i, :])
        if accept:
            history[k, i+1, :] = proposal
            acceptances[k, i] += 1
        if visualize and i % vis_every == 0 and k == 0:
            plot(curr_walker, ensemble, proposal, accept, ens_icov, ens_mean, i)
    plt.ioff()
    return history, acceptances


def plot(curr_walker, ensemble, proposal, accept, ens_icov, ens_mean, i):
    x, y = proposal if accept else curr_walker

    plt.cla()
    plt.axis([-5, 5, -5, 5])
    plt.grid()

    x_coords = np.array(np.concatenate([ensemble[:, 0], np.atleast_1d(x)])).flatten()
    y_coords = np.array(np.concatenate([ensemble[:, 1], np.atleast_1d(y)])).flatten()
    plt.scatter(x_coords, y_coords, label='ensemble')

    nth = 200
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    om = np.vstack([np.cos(th), np.sin(th)])
    uc = np.einsum('ij, ji->i', om.T, np.dot(ens_icov, om))
    r = 1. / np.sqrt(uc)
    el_x = r * om[0, :] + ens_mean[0]
    el_y = r * om[1, :] + ens_mean[1]
    plt.scatter(el_x, el_y, marker=".", alpha=0.6)

    cx = np.array([ens_mean[0]])
    cy = np.array([ens_mean[1]])
    wsize = np.array([40])
    plt.scatter(cx, cy, c='k', s=wsize, label='ensemble mean')

    plt.scatter(curr_walker[0], curr_walker[1], marker='x', c='r', s=64., label='curr_walker')
    plt.scatter(proposal[0], proposal[1], marker='x', c='g', s=64., label='proposal')

    title = "after sweep " + str(i)
    plt.title(title)
    plt.legend(loc=2)
    plt.pause(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=8000)
    parser.add_argument('--nwalkers', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--mode', type=str, default='pcn')
    parser.add_argument('--name', type=str, default='gaussian')
    parser.add_argument('--max-lag', type=int, default=1000)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vis-every', type=int, default=200)
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
    rand = np.random.RandomState()
    history, acceptances = run(args.dim, args.nwalkers, args.beta, args.niter, rand, args.visualize)

    name = '%s_nwalkers_%s_beta_%s_dim_%s.jpg' % (args.name, args.nwalkers, args.beta, args.dim)
    plot_acf(history, max_lag=args.max_lag, mean_first=True, save=name)


if __name__ == '__main__':
    main()