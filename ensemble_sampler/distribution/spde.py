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
        return - np.sum((x[1:] - x[:-1])**2 * self._N / 2.0 - (1 - (x[1:] + x[:-1])**2)**2 / (2.0 * self._N))

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N