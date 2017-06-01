import numpy as np
from emcee.autocorr import *

from history import History

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
        self._history.reset()

    def clear(self):
        self._history.clear()

    def _sample(self, niter, batch_size, verbose, print_every, store, store_every, save_dir, title, **kwargs):
        for i in xrange(niter):
            acceptances = np.empty([self.nwalkers, 1])
            lnprobs = np.empty([self.nwalkers, 1])
            curr_lnprob = None
            for j in xrange(int(np.ceil(self.nwalkers // batch_size))):
                start = (j * batch_size) % self.nwalkers
                idx = np.remainder(start + np.arange(batch_size), self.nwalkers)
                all_walkers = self._history.curr_pos

                curr_walker = all_walkers[idx]
                ens_idx = np.arange(self.nwalkers)
                ens_idx[idx] = -1
                ensemble = all_walkers[ens_idx >= 0]
                if curr_lnprob is None:
                    curr_lnprob = self.t_dist.get_lnprob(curr_walker)

                proposal = self.proposal.propose(curr_walker, ensemble, random=self._random, **kwargs)

                proposal_lnprob = self.t_dist.get_lnprob(proposal)
                ln_transition_prob_1 = self.proposal.ln_transition_prob(proposal, curr_walker)
                ln_transition_prob_2 = self.proposal.ln_transition_prob(curr_walker, proposal)

                ln_acc_prob = (proposal_lnprob + ln_transition_prob_1) - (curr_lnprob + ln_transition_prob_2)

                accept = (np.log(self._random.uniform(size=batch_size)) < np.minimum(0, ln_acc_prob))

                curr_lnprob[accept] = proposal_lnprob[accept]
                acceptances[idx] = np.array(accept, dtype=int)[:, None]
                lnprobs[idx] = ln_acc_prob[:, None]

                self._history.move(walker_idx=idx[accept], new_pos=proposal[accept])

            if verbose and i % print_every == 0:
                print '====iter %s====' % i
                print 'accepted %d proposals' % np.sum(acceptances, dtype=int)

            if store:
                itr = i if store_every is None else i % store_every
                self._history.update(itr=itr, accepted=acceptances, lnprob=lnprobs, chain=self._history.curr_pos)
                if store_every is not None and i % store_every == 0:
                    self._history.save_to(save_dir, title)
                    self._history.clear()
            else:
                yield self._history.curr_pos, lnprobs, acceptances

    def run_mcmc(self, niter, batch_size=1, p0=None, rstate0=None, verbose=False, print_every=200,
                 store=False, store_every=None, save_dir=None, title='', **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param niter:
            The number of steps to run.
        :param batch_size:
            In each iteration move `batch_size` walkers simultaneously using all other walkers as ensemble.
        :param p0:
            The initial position vector.  Can also be None to resume from where :func:``run_mcmc`` left off 
            the last time it executed.
        :param rstate0:
            The initial random state. Use default if None.
        :param verbose:
            Print information to keep track of sampling process. 
        :param print_every:
            How often to print.
        :param store:
            Store chain if True, otherwise discard chain.
        :param store_every:
            How often to save chain to file and free-up memory. If None, store the entire chain
        :param save_dir:
            The directory to save history. 
        :param title:
            Title of the saved file.
        :param kwargs:
            Optional keywords arguments for proposal / calculating lnprob.
        """
        assert self.nwalkers % batch_size == 0, 'Batch size must divide number of walkers.'
        if rstate0 is not None:
            self._random.set_state(rstate0)
        self._history.niter = niter
        self._history.save_every = niter if store_every is None else store_every

        if self._history.curr_pos is None:
            if p0 is None:
                raise ValueError("Cannot have p0=None if run_mcmc has never been called.")
            else:
                self._history.curr_pos = p0

        for h in self._sample(niter, batch_size, verbose, print_every, store, store_every, save_dir, title, **kwargs):
            pass

    def auto_corr(self, low=10, high=None, step=1, c=5, fast=False):
        """
        Adopted from emcee.ensemble. See emcee docs for detail. 
        """
        chain = self.t_dist.get_auto_corr_f(np.mean(self._history.get('chain'), axis=0))
        return integrated_time(chain, axis=0, low=low, high=high, step=step, c=c, fast=fast)

    @property
    def history(self):
        return self._history

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        self._random.set_state(state)
