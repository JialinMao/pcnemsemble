"""
Simple script for running experiments in terminals. 
Sample command: 
python test.py --distribution spde --niters 100000 --title pcn-with-ensemble-1 --s 3 --beta 0.1 --dim 50 
    --verbose --print-every 5000 --save-dir ./data/
"""

import argparse
import sys; sys.path.insert(0, '../')
import ensemble_sampler as es

SUPPORTED_DIST = ['spde', 'rosenbrock']

parser = argparse.ArgumentParser()
parser.add_argument('--distribution', type=str)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--nwalkers', type=int, default=100)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--pre', type=int, default=0)
parser.add_argument('--n', type=int, default=5)

# for proposal
parser.add_argument('--s', type=int, default=None)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--scale', type=float, default=None)
parser.add_argument('--symmetric', type=bool, default=True)
parser.add_argument('--plot', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--print-every', type=int, default=200)

parser.add_argument('--title', type=str, default=None)
parser.add_argument('--save-every', type=int, default=1)
parser.add_argument('--save-dir', type=str, default=None)

# for spde distribution
args = parser.parse_args()

dim = args.dim
batch_size = args.batch_size
nwalkers = args.nwalkers
niters = args.niters
pre = args.pre
n = args.n

if args.distribution not in SUPPORTED_DIST:
    print 'Distribution %s not supported. Supported distributions are %s.' % (args.distribution, SUPPORTED_DIST)
elif args.distribution == 'spde':
    t_dist = es.SPDE(N=dim)
elif args.distribution == 'rosenbrock':
    t_dist = es.Rosenbrock()

proposal = es.PCNWalkMove(s=args.s, beta=args.beta, scale=args.scale, symmetric=args.symmetric)
sampler = es.Sampler(dim=dim, t_dist=t_dist, proposal=proposal, nwalkers=nwalkers)

es.run(dim, sampler, batch_size, niters, n, pre, nwalkers,
       args.title, args.verbose, args.print_every, args.plot, args.save_dir, args.save_every)
