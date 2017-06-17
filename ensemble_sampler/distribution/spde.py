# NOTE: make it re-usable for more general SPDEs?
from __future__ import division
import numpy as np
from distribution import Distribution
import warnings

__all__ = ['SPDE']


class SPDE(Distribution):
    def __init__(self, N):
        """
        The invariant distribution of the stochastic Allen-Cahn equation.
        See Goodman & Weare, Ensemble Samplers With Affine Invariance for more details.
        
        :param N: the number of discretization steps to use
        """
        self._N = N
        self.dim = N
        super(SPDE, self).__init__(N=N)

    def get_lnprob(self, x):
        """
        x must be of shape [batch_size, N]
        """
        # TODO: handle overflow errors.
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                p_u_i = (x[:, 1:] - x[:, :-1])**2 * self._N / 2.0 - (1 - (x[:, 1:] + x[:, :-1])**2)**2 / (2.0 * self._N)
                return - np.sum(p_u_i, axis=1)
            except Warning as e:
                print e
                print x

    def get_auto_corr_f(self, chain):
        return np.sum(chain[:, 1:] + chain[:, :-1], axis=1) / (2.0 * self._N)
