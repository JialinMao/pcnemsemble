import numpy as np


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

