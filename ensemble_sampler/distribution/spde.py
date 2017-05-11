# NOTE: make it re-usable for more general SPDEs?
from __future__ import division
import numpy as np
from distribution import Distribution

__all__ = ['SPDE']


class SPDE(Distribution):
    def __init__(self, N):
        """
        The invariant distribution of the stochastic Allen-Cahn equation.
        See Goodman & Weare, Ensemble Samplers With Affine Invariance for more details.
        
        :param N: the number of discretization steps to use
        """
        self._N = N
        super(SPDE, self).__init__()

    def get_lnprob(self, x):
        """
        x must be of shape [batch_size, N]
        """
        p_u_i = (x[:, 1:] - x[:, :-1])**2 * self._N / 2.0 - (1 - (x[:, 1:] + x[:, :-1])**2)**2 / (2.0 * self._N)
        return - np.sum(p_u_i, axis=1)

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N