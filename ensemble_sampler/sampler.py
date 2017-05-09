import numpy as np
from emcee import autocorr

from .history import History

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
        self._history = History(dim=dim, nwalkers=nwalkers)

        self._random = np.random.RandomState()

    def reset(self):
        """
        Clear chain. 
        """
        self._history.reset()

    def _sample(self, niter, **kwargs):

        record_every = niter // self._history.max_len

        for i in range(niter):
            idx = self._random.choice(np.arange(self.nwalkers))
            all_walkers = self._history.curr_pos

            curr_walker = all_walkers[idx]
            ensemble = all_walkers
            ens_idx = np.delete(np.arange(self.nwalkers), idx)

            proposal = self.proposal.propose(curr_walker, ensemble, ens_idx=ens_idx, random=self._random, **kwargs)

            ln_acc_prob = self.t_dist.get_lnprob(proposal) + self.proposal.ln_transition_prob(proposal, curr_walker) \
                - (self.t_dist.get_lnprob(curr_walker) + self.proposal.ln_transition_prob(curr_walker, proposal))

            accept = (np.log(self._random.uniform()) < min(0, ln_acc_prob))

            if accept:
                self._history.curr_pos = proposal

            if i % record_every == 0:
                self._history.update(walker_idx=idx, accepted=accept, lnprob=ln_acc_prob)

            return self._history.history

    def run_mcmc(self, p0, niter, record_every=1, rstate0=None, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param p0:
            The initial position vector.  Can also be None to resume from
            where :func:``run_mcmc`` left off the last time it executed.
        :param niter:
            The number of steps to run.
        :param record_every:
            How often do we write to self._chain.
        :param rstate0:
            The initial random state. Use default if None.
        """
        if rstate0 is not None:
            self._random.set_state(rstate0)
        self._history.max_len = niter // record_every
        self._history.reset()

        if self._history.curr_pos is None:
            if p0 is None:
                raise ValueError("Cannot have pos0=None if run_mcmc has never "
                                 "been called.")
            else:
                self._history.curr_pos = p0

        result = self._sample(niter, **kwargs)

        return result

    def get_autocorr(self, low=10, high=None, step=1, c=10, fast=False):
        """
        Adopted from emcee.ensemble. See emcee docs for detail. 
        """
        return autocorr.integrated_time(np.mean(self._history.get("chain"), axis=0), axis=0,
                                        low=low, high=high, step=step, c=c,
                                        fast=fast)

    @property
    def history(self):
        return self._history

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        self._random.set_state(state)
