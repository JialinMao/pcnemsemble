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
        self._a = a
        self._b = b
        super(Rosenbrock, self).__init__()

    def get_lnprob(self, x):
        x_1, x_2 = x[:, 0], x[:, 1]
        return -(self._a * (x_2 - x_1 ** 2) ** 2 + (1.0 - x_1) ** 2) / self._b

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        self._b = b
