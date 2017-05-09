"""
Example of a Distribution object.
"""
import numpy as np
from .distribution import Distribution

__all__ = ['MultivariateGaussian']


class MultivariateGaussian(Distribution):
    def __init__(self, mu, cov, dim=None):
        """
        Generate a multivariate gaussian distribution object.
        
        :param mu: mean of the gaussian, of shape (dim, )
        :param cov: covariance matrix of the gaussian, of shape (dim, dim)
        :param dim (optional): dimension of the gaussian, used to check the eligibility of input parameter
        """
        assert mu.shape[0] == cov.shape[0] == cov.shape[1], "mean & covariance shape not match"
        if dim is not None:
            assert mu.shape[0] == dim, "mean & covariance does not match dimension"
        self._mu = mu
        self._cov = cov
        super(MultivariateGaussian).__init__(self, mu, cov)

    def get_lnprob(self, x):
        assert x.shape[0] == self._mu.shape[0], "input shape not match"
        diff = x - self._mu
        return -np.dot(np.dot(diff.T, self._cov), diff) / 2

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        self._mu = mu

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov):
        self._cov = cov

