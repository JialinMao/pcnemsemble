"""
The base sampler class
"""

import numpy as np

from .chain import Chain

__all__ = ["Sampler"]


class Sampler(object):

    def __init__(self, dim, t_dist, proposal, nwalkers=1):
        """
        :param dim: dimension of the sample space  
        :param nwalkers: number of walkers to use for sampling
        :param t_dist: target distribution
        :param proposal: proposal method
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

    def _sample(self, **kwargs):
        raise NotImplementedError("The acceptance probability calculation must be implemented "
                                  "by subclasses")

    def run_mcmc(self, p0, N, rstate0=None, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param p0:
            The initial position vector.  Can also be None to resume from
            where :func:``run_mcmc`` left off the last time it executed.

        :param N:
            The number of steps to run.
        """
        if rstate0 is not None:
            self._random.set_state(rstate0)
        self.chain.max_len = N

        if self.chain.curr_pos is None:
            if p0 is None:
                raise ValueError("Cannot have pos0=None if run_mcmc has never "
                                 "been called.")
            else:
                self.chain.curr_pos = p0

        self._sample(**kwargs)

        return self._chain

    @property
    def chain(self):
        return self._chain

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        self._random.set_state(state)