import numpy as np
from emcee import  autocorr

from .chain import Chain

__all__ = ["Sampler"]


class Sampler(object):

    def __init__(self, dim, t_dist, proposal, nwalkers=1):
        """
        :param dim: dimension of the sample space  
        :param t_dist: target distribution
        :param proposal: proposal method
        :param nwalkers: number of walkers to use for sampling, default set to 1 for non-ensemble methods.
        """
        self.dim = dim
        self.nwalkers = nwalkers
        self.t_dist = t_dist
        self.proposal = proposal
        self._chain = Chain(dim=dim, nwalkers=nwalkers)

        self._random = np.random.RandomState()

    def reset(self):
        """
        Clear chain. 
        """
        self.chain.reset()

    def _sample(self, N, **kwargs):

        record_every = N // self._chain.max_len

        for i in range(N):
            idx = self._random.choice(np.arange(self.nwalkers))
            all_walkers = self._chain.curr_pos

            curr_walker = all_walkers[idx]
            ensemble = all_walkers
            ens_idx = np.delete(np.arange(self.nwalkers), idx)

            proposal = self.proposal.propose(curr_walker, ensemble, ens_idx=ens_idx, random=self._random, **kwargs)

            ln_acc_prob = self.t_dist.get_lnprob(proposal) + self.proposal.ln_transition_prob(proposal, curr_walker) \
                - (self.t_dist.get_lnprob(curr_walker) + self.proposal.ln_transition_prob(curr_walker, proposal))

            accept = (np.log(self._random.uniform()) < min(0, ln_acc_prob))

            if accept:
                self._chain.curr_pos = proposal
                # TODO: chain updating.

    def run_mcmc(self, p0, N, record_every=1, rstate0=None, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param p0:
            The initial position vector.  Can also be None to resume from
            where :func:``run_mcmc`` left off the last time it executed.
        :param N:
            The number of steps to run.
        :param record_every:
            How often do we write to self._chain.
        :param rstate0:
            The initial random state. Use default if None.
        """
        if rstate0 is not None:
            self._random.set_state(rstate0)
        self._chain.max_len = N // record_every

        if self._chain.curr_pos is None:
            if p0 is None:
                raise ValueError("Cannot have pos0=None if run_mcmc has never "
                                 "been called.")
            else:
                self._chain.curr_pos = p0

        self._sample(N, **kwargs)

        return self._chain

    def get_autocorr(self, low=10, high=None, step=1, c=10, fast=False):
        """
        Adopted from emcee.ensemble. See emcee docs for detail. 
        """
        return autocorr.integrated_time(np.mean(self._chain.get("sample"), axis=0), axis=0,
                                        low=low, high=high, step=step, c=c,
                                        fast=fast)

    @property
    def chain(self):
        return self._chain

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        self._random.set_state(state)