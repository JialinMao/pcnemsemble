import numpy as np
import h5py
import argparse
import warnings


def lnprob(x):
    # x.shape = (L, n)
    return - 1.0 / 2.0 * np.einsum('ij, ji->i', x, x.T).squeeze()


def transition_ln_prob_pcn(x, y, ens_mean, ens_cov, beta):
    try:
        icov = np.linalg.inv(ens_cov)
    except np.linalg.LinAlgError, e:
        print e
        print ens_cov

    mu = ens_mean + np.sqrt(1 - beta ** 2) * (x - ens_mean)
    diff = np.expand_dims(y - mu, axis=0)
    return - 1.0 / 2.0 * np.einsum('ij, ji->i', diff, np.dot(icov, diff.T)).squeeze()


def transition_ln_prob(x, y, ens_mean, ens_cov, beta):
    return 0.0


def transition_ln_prob_test(x, y, ens_mean, ens_cov, beta):
    icov = np.linalg.inv(ens_cov)
    diff = np.expand_dims((y - np.sqrt(1 - beta ** 2) * x) / beta, axis=0)
    return - 1.0 / 2.0 * np.einsum('ij, ji->i', diff, np.dot(icov, diff.T)).squeeze()


def propose(curr_walker, ensemble, beta, dim, mode='pcn'):
    ens_mean = np.mean(ensemble, axis=0)
    ens_cov = np.atleast_2d(np.cov(ensemble.T))
    W = np.random.multivariate_normal(np.atleast_1d(np.zeros(dim)), ens_cov)
    if mode == 'pcn':
        proposal = ens_mean + np.sqrt(1 - beta ** 2) * (curr_walker - ens_mean) + beta * W
        trans_ln_prob_1 = transition_ln_prob_pcn(proposal, curr_walker, ens_mean, ens_cov, beta)
        trans_ln_prob_2 = transition_ln_prob_pcn(curr_walker, proposal, ens_mean, ens_cov, beta)
    elif mode == 'rwm':
        proposal = curr_walker + beta * W
        trans_ln_prob_1 = transition_ln_prob(proposal, curr_walker, ens_mean, ens_cov, beta)
        trans_ln_prob_2 = transition_ln_prob(curr_walker, proposal, ens_mean, ens_cov, beta)
    elif mode == 'test':
        proposal = np.sqrt(1 - beta ** 2) * curr_walker + beta * W
        trans_ln_prob_1 = transition_ln_prob_test(proposal, curr_walker, ens_mean, ens_cov, beta)
        trans_ln_prob_2 = transition_ln_prob_test(curr_walker, proposal, ens_mean, ens_cov, beta)
    return proposal, trans_ln_prob_1, trans_ln_prob_2


def accept(x, y, ln_trans_1, ln_trans_2):
    x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
    ln_accept_prob = lnprob(y) + ln_trans_1 - lnprob(x) - ln_trans_2
    return np.log(np.random.uniform()) < np.minimum(0, ln_accept_prob)


def sample(niter, p0, nwalkers, dim, beta, mode='pcn'):
    # import ipdb; ipdb.set_trace()
    # p0.shape = (L, n)
    curr_pos = p0
    curr_lnprob = lnprob(p0)
    for i in xrange(niter):
        for k in xrange(nwalkers):
            curr_walker = curr_pos[k]
            ensemble = curr_pos[np.arange(nwalkers) != k]
            proposal, trans_ln_prob_1, trans_ln_prob_2 = propose(curr_walker, ensemble, beta, dim, mode)
            ln_proposal_prob = lnprob(np.expand_dims(proposal, axis=0))
            ln_accept_prob = ln_proposal_prob + trans_ln_prob_1 - curr_lnprob[k] - trans_ln_prob_2
            accept = np.log(np.random.uniform()) < np.minimum(0, ln_accept_prob)
            if accept:
                curr_pos[k] = proposal
                curr_lnprob[k] = ln_proposal_prob
            yield proposal, accept, ln_accept_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=10000)
    parser.add_argument('--nwalkers', type=int, default=4)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--pcn', action='store_true')
    args = parser.parse_args()
    p0 = np.random.randn(args.nwalkers * args.dim).reshape([args.nwalkers, args.dim])
    for h in sample(args.niter, p0, args.nwalkers, args.dim, args.beta, args.pcn):
        proposal, accept, curr_pos = h
        print accept


if __name__ == '__main__':
    main()
