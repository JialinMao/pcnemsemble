import numpy as np
import matplotlib.pyplot as plt
import argparse
from ensemble_sampler.utils import *


def lnprob(x):
    # x.shape = (L, n)
    return -0.5 * np.einsum('ij, ji->i', x, x.T).squeeze()


def transition_ln_prob(x, y, ens_mean, icov, beta, mode):
    if mode == 'pcn':
        mu = ens_mean + np.sqrt(1 - beta ** 2) * (x - ens_mean)
    elif mode == 'old':
        mu = np.sqrt(1 - beta ** 2) * x
    else:
        return 0.0
    diff = np.expand_dims((y - mu) / beta, axis=0)
    return -0.5 * np.einsum('ij, ji->i', diff, np.dot(icov, diff.T)).squeeze()


def propose(curr_walker, ensemble, beta, dim, mode='pcn'):
    ens_mean = np.mean(ensemble, axis=0)
    ens_cov = np.atleast_2d(np.cov(ensemble.T))
    ens_icov = np.linalg.inv(ens_cov)
    W = np.random.multivariate_normal(np.atleast_1d(np.zeros(dim)), ens_cov)
    if mode == 'pcn':
        proposal = ens_mean + np.sqrt(1 - beta ** 2) * (curr_walker - ens_mean) + beta * W
    elif mode == 'old':
        proposal = np.sqrt(1 - beta ** 2) * curr_walker + beta * W
    else:
        proposal = curr_walker + beta * W
    trans_ln_prob_1 = transition_ln_prob(proposal, curr_walker, ens_mean, ens_icov, beta, mode)
    trans_ln_prob_2 = transition_ln_prob(curr_walker, proposal, ens_mean, ens_icov, beta, mode)
    return proposal, ens_icov, ens_mean, trans_ln_prob_1, trans_ln_prob_2


def sample(niter, p0, nwalkers, dim, beta, mode='pcn'):
    curr_pos = p0
    curr_lnprob = lnprob(p0)
    for i in xrange(niter):
        for k in xrange(nwalkers):
            curr_walker = curr_pos[k].copy()
            ensemble = curr_pos[np.arange(nwalkers) != k]
            proposal, ens_icov, ens_mean, trans_ln_prob_1, trans_ln_prob_2 = propose(curr_walker, ensemble, beta, dim, mode)
            ln_proposal_prob = lnprob(np.expand_dims(proposal, axis=0))
            ln_aprob = ln_proposal_prob + trans_ln_prob_1 - curr_lnprob[k] - trans_ln_prob_2
            aprob = np.exp(np.minimum(0, ln_aprob))
            accept = np.random.uniform() < aprob 
            if accept:
                curr_pos[k] = proposal
                curr_lnprob[k] = ln_proposal_prob
            yield curr_walker, ensemble, proposal, accept, ens_icov, ens_mean, i, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=8000)
    parser.add_argument('--nwalkers', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--mode', type=str, default='pcn')
    parser.add_argument('--vis-every', type=int, default=200)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--max-lag', type=int, default=1000)
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
    p0 = np.random.randn(args.nwalkers, args.dim)

    # 1st to 4th moments for each walker
    moments = np.zeros([args.nwalkers, 14])
    history = np.zeros([args.nwalkers, args.niter, args.dim])

    plt.ion()
    count = 0
    for h in sample(args.niter, p0, args.nwalkers, args.dim, args.beta, args.mode):
        curr_walker, ensemble, proposal, accept, ens_icov, ens_mean, i, k = h
        x, y = proposal if accept else curr_walker

        if k == 0:
            history[1:, i, :] = ensemble
            history[0, i, :] = x, y

        m = np.array([x, y,
                      x*x, x*y, y*y,
                      x**3, x**2*y, x*y**2, y**3,
                      x**4, x**3*y, x**2*y**2, x*y**3, y**4])
        moments[k, :] += m
        count += 1.0 / args.nwalkers

        if i % args.vis_every == 0 and k == 0:
            avg_moments = np.mean(moments / count, axis=0) 
            print 'moments after sweep %s: ' % i  
            print '(x, y): ', avg_moments[:2] 
            print '(xx, xy, yy): ', avg_moments[2:5] 
            print '(xxx, xxy, xyy, yyy): ', avg_moments[5:9] 
            print '(xxxx, xxxy, xxyy, xyyy, yyyy): ', avg_moments[9:] 

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

            title = "after sweep " + str(i) + ", with beta = " + str(args.beta)
            plt.title(title)
            plt.legend(loc=2)
            plt.pause(0.1)

    plt.ioff()
    plot_acf(history, max_lag=args.max_lag, mean_first=True, save=args.name)


if __name__ == '__main__':
    main()
