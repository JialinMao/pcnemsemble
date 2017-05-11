"""
Example of a Distribution object.
"""
import numpy as np
from distribution import Distribution

__all__ = ['MultivariateGaussian']


class MultivariateGaussian(Distribution):
    def __init__(self, mu, icov=None, cov=None, dim=None):
        """
        Generate a multivariate gaussian distribution object.
        
        :param mu: 
            mean of the gaussian, of shape (dim, )
        :param cov (optional): 
            covariance matrix of the gaussian, of shape (dim, dim). At least one of icov & cov should be given.
        :param icov (optional): 
            inverse of covariance matrix of the gaussian, of shape (dim, dim)
        :param dim (optional): 
            dimension of the gaussian, used to check the eligibility of input parameter
        """
        self._mu = mu
        self._icov = icov
        if icov is None:
            assert cov is not None, "Either covariance or inverse covariance must be given."
            self._icov = np.linalg.inv(cov)
        super(MultivariateGaussian, self).__init__()

    def get_lnprob(self, x):
        assert x.shape[1] == self._mu.shape[0], "input shape not match, %d != %d" % (x.shape[0], self._mu.shape[0])
        diff = x - self._mu
        return -np.diag(np.dot(diff, np.dot(self._icov, diff.T))) / 2.0

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        self._mu = mu

    @property
    def icov(self):
        return self._icov

    @icov.setter
    def icov(self, icov=None, cov=None):
        self._icov = icov
        if icov is None:
            assert cov is not None, "Covariance or inverse covariance must be given."
            self._icov = np.linalg.inv(cov)

