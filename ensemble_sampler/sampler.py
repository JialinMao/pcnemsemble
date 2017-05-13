import numpy as np
import timeit

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
        """
        Clear chain. 
        """
        self._history.reset()

    def run_mcmc(self, niter, batch_size=1, p0=None, rstate0=None, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param niter:
            The number of steps to run.
        :param batch_size:
            In each iteration move `batch_size` walkers simultaneously using all other walkers as ensemble.
        :param p0:
            The initial position vector.  Can also be None to resume from
            where :func:``run_mcmc`` left off the last time it executed.
        :param rstate0:
            The initial random state. Use default if None.
        :param kwargs:
            Optional keywords arguments.
             verbose: print every _print_every_ iteration. For debugging purpose.
             store: store to history every _store_every_ iterations.
        """
        if rstate0 is not None:
            self._random.set_state(rstate0)
        self._history.niter = niter
        self._history.reset()

        if self._history.curr_pos is None:
            if p0 is None:
                raise ValueError("Cannot have p0=None if run_mcmc has never "
                                 "been called.")
            else:
                self._history.curr_pos = p0

        for i in range(niter):
            acceptances= np.empty([self.nwalkers, 1])
            lnprobs = np.empty([self.nwalkers, 1])
            for j in range(int(np.ceil(self.nwalkers // batch_size))):
                start = (j * batch_size) % self.nwalkers
                idx = np.remainder(start + np.arange(batch_size), self.nwalkers)
                all_walkers = self._history.curr_pos

                curr_walker = all_walkers[idx]
                curr_lnprob = self.t_dist.get_lnprob(curr_walker)
                ensemble = all_walkers
                ens_idx = np.delete(np.arange(self.nwalkers), idx)

                proposal = self.proposal.propose(curr_walker, ensemble, ens_idx=ens_idx, random=self._random, **kwargs)

                ln_acc_prob = self.t_dist.get_lnprob(proposal) + self.proposal.ln_transition_prob(proposal, curr_walker) \
                              - (curr_lnprob + self.proposal.ln_transition_prob(curr_walker, proposal))

                accept = (np.log(self._random.uniform(size=kwargs.get('sample_size', 1))) < np.minimum(0, ln_acc_prob))

                acceptances[idx] = np.array(accept, dtype=int)[:, None]
                lnprobs[idx] = ln_acc_prob[:, None]

                self._history.move(walker_idx=idx[accept], new_pos=proposal[accept])

            if kwargs.get('verbose', False) and i % kwargs.get('print_every', 200) == 0:
                print '====iter %s====' % i
                print 'accept', acceptances

            if kwargs.get('store', True) and i % kwargs.get('store_every', 1) == 0:
                self._history.update(itr=i, accepted=acceptances, lnprob=lnprobs, chain=self._history.curr_pos)

        return self._history

    @property
    def history(self):
        return self._history

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        self._random.set_state(state)
