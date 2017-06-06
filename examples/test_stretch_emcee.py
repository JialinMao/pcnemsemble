import sys; sys.path.insert(0, "/Users/M/emcee")
import emcee
import numpy as np

nwalkers = 250
dim = 2
N = 10000


class Rosenbrock(object):
    def __init__(self):
        self.a1 = 100.0
        self.a2 = 20.0

    def __call__(self, p):
        return -(self.a1 * (p[1] - p[0] ** 2) ** 2 + (1 - p[0]) ** 2) / self.a2

p0 = np.random.rand(nwalkers * dim).reshape(nwalkers, dim)
sampler = emcee.EnsembleSampler(nwalkers, dim, Rosenbrock())
hist = sampler.run_mcmc(p0, N=N)
