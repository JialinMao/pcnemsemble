import sys; sys.path.insert(0, '../')
import numpy as np
from ensemble_sampler import *

dim = 2
nwalkers = 5
niters = 10000

t_dist = Rosenbrock()
proposal = StretchMove(a=2.0)
sampler = Sampler(dim=dim, t_dist=t_dist, proposal=proposal, nwalkers=nwalkers)
sampler.run_mcmc(niters, p0=np.random.randn(dim*nwalkers).reshape([nwalkers, dim]), store=True)


