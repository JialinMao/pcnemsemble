"""
Simple script for running experiments in terminals. 
"""
import os
import sys; sys.path.insert(0, '../')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ensemble_sampler as es
from emcee.autocorr import *
import emcee

SUPPORTED_DIST = ['gaussian', 'fitting', 'exponential']
SUPPORTED_PROPOSAL = ['PCNWalk', 'Gaussian', 'Walk', 'Stretch', 'Stretch_emcee']


def main():
    parser = argparse.ArgumentParser()
    # distribution options
    parser.add_argument('--distribution', type=str, required=True)
    parser.add_argument('--dim', type=int, default=2)

    # sampler options
    parser.add_argument('--nwalkers', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--niters', type=int, default=10000)

    # proposal options
    parser.add_argument('--mode', nargs='+', type=str, default=None)
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--a', type=float, default=2.0)

    # some args for posterior fitting distribution
    parser.add_argument('--prior', type=str, default='FlatPrior')
    parser.add_argument('--prior-args', nargs='+', type=str, default=None)

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

    if args.distribution == 'fitting':
        dim = args.dim * 2 + 1
    else:
        dim = args.dim

    n_ = args.steps if args.vary is not None else 1
    proposals = SUPPORTED_PROPOSAL if args.mode is None else args.mode
    k_ = len(proposals)

    values = {'nwalkers': args.nwalkers * np.ones(n_, dtype=int),
              'beta': args.beta * np.ones(n_),
              'dim': dim * np.ones(n_, dtype=int),
              'a': args.a * np.ones(n_)}

    if args.vary == 'nwalkers':
        values['nwalkers'] = args.dim * 4 + args.gap * np.arange(n_)
    elif args.vary == 'beta':
        values['beta'] = np.linspace(0, 1, n_ + 2)[1:-1]
    elif args.vary == 'dim':
        values['dim'] = np.concatenate([[2], np.arange(1, n_) * args.gap])
        values['nwalkers'] = args.ens_dim_ratio * values['dim']

    print values

    act_info = np.zeros([k_, n_, 2])
    accept_info = np.zeros([k_, n_])

    # iterate through all 3 sampling methods
    for k in range(k_):
        for i in range(n_):
            p = proposals[k]
            dim = values['dim'][i] * 2 + 1 if args.distribution == 'fitting' else values['dim'][i]
            nwalkers = values['nwalkers'][i]
            beta = values['beta'][i]
            a = values['a'][i]

            niters = args.niters
            batch_size = args.batch_size or nwalkers // 2

            p_args = dict(beta=beta, a=a)
            if p != 'Stretch_emcee':
                proposal = getattr(es, p)(**p_args)

            assert args.distribution in SUPPORTED_DIST, \
                'Distribution %s not supported. Supported distributions are %s.' % (args.distribution, SUPPORTED_DIST)

            p0 = np.random.randn(nwalkers, dim)
            if args.distribution == 'gaussian':
                t_dist = es.MultivariateGaussian(mu=np.zeros(dim), cov=np.identity(dim))

            elif args.distribution == 'fitting':
                m = (dim - 1) / 2
                A = np.ones(m)
                l = np.ones(m) + 0.1 * np.arange(m)
                s = 1
                print 'A=%s, l=%s' % (A, l)

                t = np.linspace(start=0, stop=1, num=10)
                data = es.generate_fake_data(A, l, s, t)
                prior = getattr(es, args.prior)() if args.prior_args is None else getattr(es, args.prior)(*args.prior_args)
                t_dist = es.LogPosterior(prior, data['t'], data['Y'], dim)
                p0 += 1.0

            elif args.distribution == 'exponential':
                t_dist = es.Exponential(dim=dim)
                p0 = np.abs(p0)

            hist_title = 'n_%s_dim_%s_b_%s_p_%s_t_dist_%s' % (nwalkers, dim, beta, p, args.distribution)

            data_dir = args.save_dir + '/data'
            if p == 'Stretch_emcee':
                sampler = emcee.EnsembleSampler(nwalkers, dim, t_dist.get_lnprob)
                h = sampler.run_mcmc(p0, niters)
                accept_info[k, i] = np.mean(sampler.acceptance_fraction) * 100
            else:
                sampler = es.Sampler(t_dist=t_dist, proposal=proposal, nwalkers=nwalkers)
                sampler.run_mcmc(niters, batch_size=batch_size, p0=p0, store_every=args.store_every,
                                 save_dir=data_dir, title=hist_title, debug=args.debug)
                accept_info[k, i] = sampler.acceptance_rate.mean()*100
            print 'avg acceptance rate: %.2f%s' % (accept_info[k, i], '%')
            try:
                if p == 'Stretch_emcee':
                    act = sampler.acor
                else:
                    act = sampler.auto_corr()
                act_mean = act.mean()
                act_std = np.sqrt(act.var())
                act_info[k, i] = [act_mean, act_std]
                print 'auto-correlation time: %s' % act
                print 'mean act: %s' % act_mean
                print 'standard deviation: %s' % act_std
            except AutocorrError, err:
                print err

    col = values[args.vary] if args.vary is not None else [''.join(['%s_%s_' % (k, values[k][0]) for k in values.keys()])]
    df_act = pd.DataFrame(act_info.reshape([k_, -1]), index=proposals,
                          columns=pd.MultiIndex.from_tuples(zip(np.repeat(col, 2), np.array(['mean', 'cov'] * n_))))
    df_accept = pd.DataFrame(accept_info, index=proposals, columns=col)
    f_dir = args.save_dir + '/plot_info'
    f_name = args.vary+args.suffix if args.vary is not None else args.suffix
    import ipdb; ipdb.set_trace()
    df_act.to_pickle(os.path.join(f_dir, 'act%s.pkl' % f_name))
    df_accept.to_pickle(os.path.join(f_dir, 'accept%s.pkl' % f_name))

    if args.plot:
        assert args.vary is not None, 'Nothing to plot'
        if args.vary == 'nwalkers':
            fig_title = "beta=%s, dim=%s, changing number of walkers" % (args.beta, args.dim)
        elif args.vary == 'beta':
            fig_title = "nwalkers=%s, dim=%s, changing beta" % (args.nwalkers, args.dim)
        elif args.vary == 'dim':
            fig_title = "nwalkers / dim=%s, beta=%s, changing dimension" % (args.ens_dim_ratio, args.beta)

        fig = plt.figure(facecolor='white')
        for i in range(k_):
            if args.mode is None:
                p = SUPPORTED_PROPOSAL[i]
            else:
                p = args.mode[i]
            idx = act_info[i, :, 0] != 0
            if sum(idx) > 0:
                plt.errorbar(values[args.vary][idx], act_info[i, :, 0][idx], yerr=act_info[i, :, 1][idx],
                             capsize=8, fmt='-o', label=p)

        plt.grid(False)
        plt.xlabel(args.vary)
        plt.ylabel('mean_act')
        plt.title(fig_title)
        plt.legend()
        pic_dir = args.save_dir + '/pics'
        plt.savefig(os.path.join(pic_dir, args.vary+'%s.jpg' % args.suffix),
                    facecolor='white', edgecolor='k')
        plt.close()

        if args.plot_accept:
            for i in range(k_):
                p = args.mode or SUPPORTED_PROPOSAL[i]
                plt.plot(values[args.vary], accept_info[i, :], label=p)

            plt.grid(False)
            plt.xlabel(args.vary)
            plt.ylabel('avg_accept_rate')
            plt.title(fig_title)
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, args.vary+'%s%s.jpg' % ('_acceptance', args.suffix)),
                        dpi=fig.dpi, facecolor='white', edgecolor='k', )

if __name__ == '__main__':
    main()

