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
        self.dim = dim or len(mu)
        self._mu = mu
        self._icov = icov
        if icov is None:
            assert cov is not None, "Either covariance or inverse covariance must be given."
            self._icov = np.linalg.inv(cov)
        super(MultivariateGaussian, self).__init__(mu=mu, icov=icov)

    def get_lnprob(self, x):
        diff = x - self._mu
        # return -np.diag(np.dot(diff, np.dot(self._icov, diff.T))) / 2.0
        return - np.einsum('ij, ji->i', diff, np.dot(self._icov, diff.T)) / 2.0
