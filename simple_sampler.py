import numpy as np
import matplotlib.pyplot as plt
import argparse


def lnprob(x):
    # x.shape = (L, n)
    return -0.5 * np.einsum('ij, ji->i', x, x.T).squeeze()


def transition_ln_prob(x, y, ens_mean, icov, beta, mode):
    if mode == 'pcn_1':
        mu = x + np.sqrt(1 - beta ** 2) * (ens_mean - x)
        diff = np.expand_dims(y - mu, axis=0)
        return -0.5 * np.einsum('ij, ji->i', diff, np.dot(icov, diff.T)).squeeze()
    elif mode == 'pcn_2':
        mu = ens_mean + np.sqrt(1 - beta ** 2) * (x - ens_mean)
        diff = np.expand_dims(y - mu, axis=0)
        return -0.5 * np.einsum('ij, ji->i', diff, np.dot(icov, diff.T)).squeeze()
    elif mode == 'old':
        diff = np.expand_dims((y - np.sqrt(1 - beta ** 2) * x) / beta, axis=0)
        return -0.5 * np.einsum('ij, ji->i', diff, np.dot(icov, diff.T)).squeeze()
    else:
        return 0.0


def propose(curr_walker, ensemble, beta, dim, mode='pcn'):
    ens_mean = np.mean(ensemble, axis=0)
    ens_cov = np.atleast_2d(np.cov(ensemble.T))
    ens_icov = np.linalg.inv(ens_cov)
    W = np.random.multivariate_normal(np.atleast_1d(np.zeros(dim)), ens_cov)
    if mode == 'pcn':
        proposal = ens_mean + np.sqrt(1 - beta ** 2) * (curr_walker - ens_mean) + beta * W
        trans_ln_prob_1 = transition_ln_prob(proposal, curr_walker, ens_mean, ens_icov, beta, 'pcn_2')
        trans_ln_prob_2 = transition_ln_prob(curr_walker, proposal, ens_mean, ens_icov, beta, 'pcn_2')
    else:
        if mode == 'old':
            proposal = np.sqrt(1 - beta ** 2) * curr_walker + beta * W
        else:
            proposal = curr_walker + beta * W
        trans_ln_prob_1 = transition_ln_prob(proposal, curr_walker, ens_mean, ens_icov, beta, mode)
        trans_ln_prob_2 = transition_ln_prob(curr_walker, proposal, ens_mean, ens_icov, beta, mode)
    return proposal, ens_icov, ens_mean, trans_ln_prob_1, trans_ln_prob_2


def sample(niter, p0, nwalkers, dim, beta, mode='pcn'):
    # import ipdb; ipdb.set_trace()
    # p0.shape = (L, n)
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
            yield proposal, accept, curr_pos, curr_walker, ens_icov, ens_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=8000)
    parser.add_argument('--nwalkers', type=int, default=4)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--mode', type=str, default='pcn')
    parser.add_argument('--vis-every', type=int, default=2000)
    args = parser.parse_args()

    np.random.seed(args.seed)
    p0 = np.random.randn(args.nwalkers, args.dim)

    i = 0
    plt.ion()
    for h in sample(args.niter, p0, args.nwalkers, args.dim, args.beta, args.mode):
        proposal, accept, curr_pos, curr_walker, ens_icov, ens_mean = h
        n = i // args.nwalkers
        k = i % args.nwalkers
        if (n % args.vis_every) == 0 and k == 0:
            plt.cla()
            plt.axis([-5, 5, -5, 5])
            plt.grid()

            x_coords = np.array(curr_pos[:, 0]).flatten()
            y_coords = np.array(curr_pos[:, 1]).flatten()
            plt.scatter(x_coords, y_coords, label='walkers')

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

            x = curr_walker.squeeze()
            y = proposal.squeeze()
            plt.scatter(x[0], x[1], marker='x', c='r', s=64., label='x')
            plt.scatter(y[0], y[1], marker='x', c='g', s=64., label='y')

            title = "after sweep " + str(n) + ", with beta = " + str(args.beta)
            plt.title(title)
            plt.legend(loc=2)
            plt.pause(0.05)
        i += 1
    

if __name__ == '__main__':
    main()
