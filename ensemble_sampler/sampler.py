import os
import numpy as np
from emcee.autocorr import *

from history import History
from utils import *

__all__ = ["Sampler"]


class Sampler(object):

    def __init__(self, t_dist, proposal, nwalkers=1):
        """
        :param t_dist: target distribution
        :param proposal: proposal method
        :param nwalkers: number of walkers to use for sampling, default set to 1 for non-ensemble methods.
        """
        self.nwalkers = nwalkers
        self.t_dist = t_dist
        self.dim = t_dist.dim
        self.proposal = proposal

        self._history = History(dim=self.dim, nwalkers=nwalkers)
        self._random = np.random.RandomState()

    def sample(self, niter, batch_size=None, p0=None, rstate0=None, store_chain=True,
               save_to_f=False, save_every=None, save_dir=None, title=None, **kwargs):
        """
        If store_chain == True, return whole history after the run is over, save to hdf5 file every `store_every`
        iterations if `save` is True.
        If store_chain == False, yields position, ln_prob and accepted or not every iteration.
        """
        if kwargs.get('debug', False):
            import ipdb; ipdb.set_trace()
        if rstate0 is not None:
            self._random.set_state(rstate0)

        batch_size = self.nwalkers // 2 if batch_size is None else batch_size
        assert self.nwalkers % batch_size == 0, 'Batch size must divide number of walkers.'

        store_every = save_every or niter
        self._history.niter = niter
        self._history.max_len = store_every

        if save_to_f:
            remove_f(title, save_dir)  # Remove file if already exist

        if self._history.curr_pos is None:
            if kwargs.get('random_start'):
                p0 = np.random.randn(self.nwalkers, self.dim)
            if p0 is None:
                raise ValueError("Cannot have p0=None if run_mcmc has never been called. "
                                 "Set `random_start=True` in kwargs to use random start")
            else:
                self._history.curr_pos = p0

        all_walkers = self._history.curr_pos  # (nwalkers, dim)
        ln_probs = self.t_dist.get_lnprob(all_walkers)  # (nwalkers, )

        n = self.nwalkers // batch_size
        for i in xrange(niter):
            if store_chain:
                self._history.update(itr=i, chain=self._history.curr_pos)
            for k in xrange(n):
                # pick walkers to move
                idx = slice(k * batch_size, (k+1) * batch_size)
                curr_walker = all_walkers[idx].copy()  # (batch_size, dim)
                if k == 0:
                    ensemble = all_walkers[batch_size:]
                elif k == n-1:
                    ensemble = all_walkers[:k * batch_size]
                else:
                    ensemble = np.vstack([all_walkers[:k*batch_size], all_walkers[(k+1)*batch_size:]])

                # propose a move
                proposal, blob = self.proposal.propose(curr_walker, ensemble, self._random, **kwargs)
                if blob is not None and store_chain and i == 0 and k == 0:
                    name_to_dim = dict([(key, blob[key].shape[1]) for key in blob.keys()])
                    self._history.add_extra(name_to_dim)
                    if store_chain:
                        self._history.update(i, idx, **blob)

                # calculate acceptance probability
                curr_lnprob = ln_probs[idx]  # (batch_size, )
                proposal_lnprob = self.t_dist.get_lnprob(proposal)
                ln_transition_prob_1 = self.proposal.ln_transition_prob(proposal, curr_walker)
                ln_transition_prob_2 = self.proposal.ln_transition_prob(curr_walker, proposal)
                proposal_lnprob[(proposal_lnprob == -np.inf) * (curr_lnprob == - np.inf)] = 0
                curr_lnprob[(proposal_lnprob == -np.inf) * (curr_lnprob == - np.inf)] = 0
                ln_acc_prob = (proposal_lnprob + ln_transition_prob_1) - (curr_lnprob + ln_transition_prob_2)

                # accept or reject
                accept = (self._random.uniform(size=batch_size) < np.exp(np.minimum(0, ln_acc_prob)))
                if store_chain:
                    self._history.update(i, idx, accepted=np.expand_dims(accept, 1))
                else:
                    yield curr_walker, ensemble, proposal, accept

                # update history records for next iteration
                self._history.move(walker_idx=np.arange(idx.start, idx.stop)[accept], new_pos=proposal[accept])
                all_walkers = self._history.curr_pos
                ln_probs[idx][accept] = proposal_lnprob[accept]

            if save_to_f and (i+1) % store_every == 0:
                try:
                    self._history.save_to(save_dir, title)
                except TypeError:
                    pass

    def run_mcmc(self, niter, **kwargs):
        for h in self.sample(niter, **kwargs):
            pass

    def auto_corr(self, low=10, high=None, step=1, c=10, fast=False):
        """
        Adopted from emcee.ensemble. See emcee docs for detail. 
        """
        chain = self.t_dist.get_auto_corr_f(np.mean(self.chain, axis=0))
        return integrated_time(chain, axis=0, low=low, high=high, step=step, c=c, fast=fast)

    @property
    def history(self):
        return self._history

    @property
    def chain(self):
        return self._history.get('chain')

    @property
    def acceptance_rate(self):
        return np.sum(self._history.get('accepted'), axis=1) / float(self._history.niter)

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        self._random.set_state(state)
