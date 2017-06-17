"""
Rosenbrock distribution.
"""
from __future__ import division
from distribution import Distribution

__all__ = ['Rosenbrock']


class Rosenbrock(Distribution):
    def __init__(self, a=100.0, b=20.0):
        """
        Generate a Rosenbrock distribution object.
        f(x_1, x_2) \propto exp(-[a(x_2 - x_1^2)^2 + (1-x_1)^2] / b)
        """
        self.a = a
        self.b = b
        self.dim = 2
        super(Rosenbrock, self).__init__(a=a, b=b)

    def get_lnprob(self, x):
        x_1, x_2 = x[:, 0], x[:, 1]
        return -(self.a * (x_2 - x_1 ** 2) ** 2 + (1.0 - x_1) ** 2) / self.b

    def get_auto_corr_f(self, chain):
        return chain

