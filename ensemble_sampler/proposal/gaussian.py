"""
Isotropic Gaussian proposal, with scale `beta`
"""
import numpy as np
from proposal import Proposal

__all__ = ['Gaussian']


class Gaussian(Proposal):

    def __init__(self, beta=0.4):
        assert 0 <= beta <= 1, "beta must be in [0, 1]"
        self.beta = beta
        super(Proposal, self).__init__()

    def propose(self, walkers_to_move, ensemble, random=None, *args, **kwargs):
        """
        walkers_to_move.shape = (batch_size, dim)
        ensemble.shape = (Nc, dim)
        """
        rand = np.random.RandomState() if random is None else random
        beta = kwargs.get('beta', self.beta)

        B, _ = walkers_to_move.shape
        Nc, dim = ensemble.shape

        W = rand.multivariate_normal(mean=np.zeros(dim), cov=np.identity(dim), size=B)
        proposal = walkers_to_move + beta * W

        return proposal, None

