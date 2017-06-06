import os
import numpy as np
from emcee.autocorr import *

from history import History

__all__ = ["Sampler"]


class Sampler(object):

    def __init__(self, dim, t_dist, proposal, nwalkers=1, debug=False):
        """
        :param dim: dimension of the sample space  
        :param t_dist: target distribution
        :param proposal: proposal method
        :param nwalkers: number of walkers to use for sampling, default set to 1 for non-ensemble methods.
        """
        self.debug = debug
        self.dim = dim
        self.nwalkers = nwalkers
        self.t_dist = t_dist
        self.proposal = proposal
        self._history = History(dim=dim, nwalkers=nwalkers)
        self._acceptances = np.zeros([self.nwalkers, ])
        self._random = np.random.RandomState()

    def init(self):
        """
        Initialize storage in self._history. 
        """
        self._history.init()

    def clear(self):
        """
        Maintain shape of stored history, reset all to 0.
        """
        self._history.clear()

    def sample(self, niter, batch_size, verbose=False, print_every=1000, store=False,
               store_every=None, save_dir='./result/', title=None, **kwargs):

        if batch_size is None:
            batch_size = self.nwalkers // 2
        assert self.nwalkers % batch_size == 0, 'Batch size must divide number of walkers.'

        try:
            os.remove(os.path.join(save_dir, title + '.hdf5'))
        except (OSError, AttributeError):
            pass

        all_walkers = self._history.curr_pos  # (nwalkers, dim)
        ln_probs = self.t_dist.get_lnprob(all_walkers)  # (nwalkers, )

        n = self.nwalkers // batch_size
        debug = False
        for i in xrange(niter):
            if i >= niter / 2 and self.debug:
                import ipdb
                ipdb.set_trace()
                debug = True

            for k in range(n):
                idx = slice(k * batch_size, (k+1) * batch_size)

                curr_walker = all_walkers[idx]  # (batch_size, dim)
                curr_lnprob = ln_probs[idx]  # (batch_size, dim)

                ens_idx = np.zeros(self.nwalkers)
                ens_idx[idx] = -1
                ensemble = all_walkers[ens_idx >= 0]  # (Nc = nwalkers-batch_size, dim)

                proposal = self.proposal.propose(curr_walker, ensemble, self._random, debug=debug, **kwargs)  # (batch_size, dim)

                proposal_lnprob = self.t_dist.get_lnprob(proposal)
                ln_transition_prob_1 = self.proposal.ln_transition_prob(proposal, curr_walker)
                ln_transition_prob_2 = self.proposal.ln_transition_prob(curr_walker, proposal)
                ln_acc_prob = (proposal_lnprob + ln_transition_prob_1) - (curr_lnprob + ln_transition_prob_2)  # (batch_size, )

                accept = (np.log(self._random.uniform(size=batch_size)) < ln_acc_prob)  # (batch_size, )

                self._acceptances[idx] += accept
                ln_probs[idx][accept] = proposal_lnprob[accept]

                self._history.move(walker_idx=np.arange(idx.start, idx.stop)[accept], new_pos=proposal[accept])
                all_walkers = self._history.curr_pos

            if verbose and i % print_every == 0:
                print '====iter %s====' % i
                print 'Acceptance probability: ' + str(np.exp(ln_acc_prob))
                print 'Accepted: ' + str(accept)
                # print 'All walkers: ' + str(all_walkers)
                # print 'ln_probs: ' + str(ln_probs)

            if store:
                if title is None:
                    from datetime import datetime
                    title = datetime.now().strftime('%Y-%m-%d_%H:%M')
                itr = i if store_every is None else i % store_every
                self._history.update(itr=itr, chain=self._history.curr_pos)
                if store_every is not None and i % store_every == 0:
                    self._history.save_to(save_dir, title)
                    self._history.clear()
            else:
                yield self._history.curr_pos, ln_probs, self._acceptances

    def run_mcmc(self, niter, batch_size=None, p0=None, rstate0=None, verbose=False, print_every=200,
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
            Store chain if True, otherwise discard chain and yield results every iteration.
        :param store_every:
            How often to save chain to file and free-up memory. Store the entire chain if None.
        :param save_dir:
            The directory to save history. 
        :param title:
            Title of the saved file.
        :param kwargs:
            Optional keywords arguments for proposal / calculating lnprob.
        """
        if batch_size is None:
            batch_size = self.nwalkers // 2

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

        for h in self.sample(niter, batch_size, verbose, print_every, store, store_every, save_dir, title, **kwargs):
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

    @property
    def acceptance(self):
        return self._acceptances
