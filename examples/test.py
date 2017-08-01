"""
Simple script for running experiments in terminals. 
"""

import argparse
import time
import sys; sys.path.insert(0, '../')
import numpy as np
import ensemble_sampler as es
from emcee.autocorr import *

SUPPORTED_DIST = ['spde', 'rosenbrock', 'gaussian']
SUPPORTED_PROPOSAL = ['walk', 'stretch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', type=str)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--nwalkers', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--niters', type=int, default=10000)

    # for proposal
    parser.add_argument('--proposal', type=str, default='walk')
    parser.add_argument('--a', type=int, default=2.0)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--print-every', type=int, default=200)

    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default=None)

    parser.add_argument('--store-every', type=int, default=None)
    parser.add_argument('--store', action='store_true')
    args = parser.parse_args()

    dim = args.dim
    batch_size = args.batch_size or args.nwalkers // 2
    nwalkers = args.nwalkers
    niters = args.niters

    assert args.distribution in SUPPORTED_DIST, 'Distribution %s not supported. Supported distributions are %s.' % (args.distribution, SUPPORTED_DIST)
    if args.distribution == 'spde':
        t_dist = es.SPDE(N=dim)
    elif args.distribution == 'rosenbrock':
        t_dist = es.Rosenbrock()
    elif args.distribution == 'gaussian':
        t_dist = es.MultivariateGaussian(mu=np.zeros(dim), cov=np.identity(dim))

    assert args.proposal in SUPPORTED_PROPOSAL, 'Distribution %s not supported. Supported distributions are %s.' % (args.proposal, SUPPORTED_PROPOSAL)
    if args.proposal == 'walk':
        proposal = es.PCNWalkMove(beta=args.beta)
    elif args.proposal == 'stretch':
        proposal = es.StretchMove(a=args.a)

    sampler = es.Sampler(t_dist=t_dist, proposal=proposal, nwalkers=nwalkers)
    start = time.time()
    sampler.run_mcmc(niters, batch_size=batch_size, random_start=True, store=True, store_every=args.save_every, title=args.title)
    end = time.time()
    print 'finishes in about %.2f seconds' % float(end - start)
    try:
        print 'auto-correlation time: %s' % sampler.auto_corr()
    except AutocorrError, err:
        print err


if __name__ == '__main__':
    main()

