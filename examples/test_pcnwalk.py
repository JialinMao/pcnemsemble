import sys; sys.path.insert(0, '../')
import argparse
from ensemble_sampler import *

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--nwalkers', type=int, default=4)
parser.add_argument('--niters', type=int, default=10000)

parser.add_argument('--s', type=int, default=3)
parser.add_argument('--beta', type=float, default=0.4)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--print-every', type=int, default=1000)

args = parser.parse_args()

print args


dim = args.dim
nwalkers = args.nwalkers
niters = args.niters

s = args.s
beta = args.beta

mu = np.random.randn(dim)

cov = 0.5 - np.random.rand(dim ** 2).reshape((dim, dim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)

print 'mu: ', mu
print 'cov: ', cov

t_dist = MultivariateGaussian(cov=cov, mu=mu, dim=dim)

proposal = WalkMove(s=s, beta=beta)
sampler = Sampler(dim=dim, t_dist=t_dist, proposal=proposal, nwalkers=nwalkers)
sampler.run_mcmc(niters, batch_size=1, p0=np.random.randn(dim*nwalkers).reshape([nwalkers, dim]),
                 verbose=args.verbose, print_every=args.print_every)
