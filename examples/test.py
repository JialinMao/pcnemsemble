"""
Simple script for running experiments in terminals. 
"""
import os
import sys; sys.path.insert(0, '../')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import ensemble_sampler as es
import emcee
from emcee.autocorr import *

SUPPORTED_DIST = ['gaussian', 'fitting', 'exponential']
SUPPORTED_PROPOSAL = ['pcn', 'ensemble', 'gaussian', 'stretch', 'emcee']


def main():
    parser = argparse.ArgumentParser()
    # distribution options
    parser.add_argument('--distribution', type=str, default='gaussian')
    parser.add_argument('--dim', type=int, default=2)

    # args for posterior distribution
    parser.add_argument('--param-scale', nargs='+', type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--t-max', type=float, default=1.0)

    # sampler options
    parser.add_argument('--nwalkers', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--niters', type=int, default=10000)

    # proposal options
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--beta', type=float, default=0.6)

    # saving options
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--store-every', type=int, default=None)

    # experimenting
    parser.add_argument('--vary', type=str, default=None)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--gap', type=int, default=10)
    parser.add_argument('--ens-dim-ratio', type=int, default=2)

    # plotting
    parser.add_argument('--plot-accept', action='store_true')
    parser.add_argument('--plot', action='store_true')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    n_ = args.steps if args.vary is not None else 1
    # test on all supported proposal if none is specified
    k_ = len(SUPPORTED_PROPOSAL) if args.mode is None else 1

    values = {'nwalkers': args.nwalkers * np.ones(n_, dtype=int),
              'beta': args.beta * np.ones(n_),
              'dim': args.dim * np.ones(n_, dtype=int)}

    if args.vary == 'nwalkers':
        values['nwalkers'] = args.dim * 4 + args.gap * np.arange(n_)
    elif args.vary == 'beta':
        values['beta'] = np.linspace(0, 1, n_ + 2)[1:-1]
    elif args.vary == 'dim':
        values['dim'] = np.concatenate([[2], np.arange(1, n_) * args.gap])
        values['nwalkers'] = args.ens_dim_ratio * values['dim']

    print values

    act_info = np.zeros([3, n_, 2])
    accept_info = np.zeros([3, n_])

    # iterate through all 3 sampling methods
    for k in range(k_):
        for i in range(n_):
            p = args.mode or SUPPORTED_PROPOSAL[k]
            if args.distribution == 'fitting':
                dim = values['dim'][i] * 2 + 1
            else:
                dim = values['dim'][i]
            nwalkers = values['nwalkers'][i]
            beta = values['beta'][i]
            niters = args.niters
            batch_size = args.batch_size or nwalkers // 2
            proposal = es.PCNWalkMove(beta=beta, mode=p)

            assert args.distribution in SUPPORTED_DIST, \
                'Distribution %s not supported. Supported distributions are %s.' % (args.distribution, SUPPORTED_DIST)

            p0 = np.random.randn(nwalkers, dim)
            if args.distribution == 'gaussian':
                t_dist = es.MultivariateGaussian(mu=np.zeros(dim), cov=np.identity(dim))

            elif args.distribution == 'fitting':
                m = (dim - 1) / 2
                A = args.param_scale[0] * np.ones(m)
                l = args.param_scale[1] * np.ones(m)
                s = args.param_scale[2]
                print 'A=%s, l=%s' % (A, l)

                t = np.linspace(start=0, stop=args.t_max, num=args.n)

                data = es.generate_fake_data(A, l, s, t)
                t_dist = es.LogPosterior(es.FlatPrior(50, 50, 50), data['t'], data['Y'], dim)

            elif args.distribution == 'exponential':
                t_dist = es.Exponential(dim=dim)
                p0 = np.abs(p0)

            hist_title = 'nwalkers_%s_dim_%s_beta_%s_mode_%s_dist_%s' % (nwalkers, dim, beta, p, args.distribution)

            data_dir = args.save_dir + '/data'
            sampler = es.Sampler(t_dist=t_dist, proposal=proposal, nwalkers=nwalkers)
            sampler.run_mcmc(niters, batch_size=batch_size, p0=p0, store=True, store_every=args.store_every,
                             save_dir=data_dir, title=hist_title, debug=args.debug)
            accept_info[k, i] = sampler.acceptance_rate.mean()*100
            print 'avg acceptance rate: %.2f%s' % (accept_info[k, i], '%')
            try:
                act = sampler.auto_corr()
                act_mean = act.mean()
                act_std = np.sqrt(act.var())
                act_info[k, i] = [act_mean, act_std]
                print 'auto-correlation time: %s' % act
                print 'mean act: %s' % act_mean
                print 'standard deviation: %s' % act_std
            except AutocorrError, err:
                print err

    if args.plot:
        assert args.vary is not None, 'Nothing to plot'
        if args.vary == 'nwalkers':
            fig_title = "beta=%s, dim=%s, changing number of walkers" % (args.beta, args.dim)
        elif args.vary == 'beta':
            fig_title = "nwalkers=%s, dim=%s, changing beta" % (args.nwalkers, args.dim)
        elif args.vary == 'dim':
            fig_title = "nwalkers / dim=%s, beta=%s, changing dimension" % (args.ens_dim_ratio, args.beta)

        for i in range(k_):
            p = args.mode or SUPPORTED_PROPOSAL[i]
            idx = act_info[i, :, 0] != 0
            if sum(idx) > 0:
                plt.errorbar(values[args.vary][idx], act_info[i, :, 0][idx], yerr=act_info[i, :, 1][idx],
                             fmt='-o', label=p)

        plt.xlabel(args.vary)
        plt.ylabel('mean_act')
        plt.title(fig_title)
        plt.legend()
        pic_dir = args.save_dir + '/pics'
        plt.savefig(os.path.join(pic_dir, args.vary+'%s.jpg' % args.suffix))
        plt.close()

        if args.plot_accept:
            for i in range(k_):
                p = args.mode or SUPPORTED_PROPOSAL[i]
                plt.plot(values[args.vary], accept_info[i, :], label=p)
            plt.xlabel(args.vary)
            plt.ylabel('avg_accept_rate')
            plt.title(fig_title)
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, args.vary+'%s%s.jpg' % ('_acceptance', args.suffix)))


if __name__ == '__main__':
    main()

