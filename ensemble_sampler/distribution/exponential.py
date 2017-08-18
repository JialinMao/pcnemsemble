from __future__ import division
import numpy as np
from distribution import Distribution


class Exponential(Distribution):
    def __init__(self, dim):
        """
        Generate a Rosenbrock distribution object.
        f(x_1, x_2) \propto exp(-[a(x_2 - x_1^2)^2 + (1-x_1)^2] / b)
        """
        self.dim = dim
        super(Exponential, self).__init__(dim=dim)

    def get_lnprob(self, x):
        prob = -np.sum(x, axis=1)
        prob[np.any(x < 0, axis=1)] = -np.inf
        return prob

