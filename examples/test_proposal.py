import sys; sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import ensemble_sampler as es

NWALKERS = 3
NITERS = 1000000
DIM = 1
BETA = 0.4
P0 = np.array([[-1], [0], [1]])


def main():
    proposals = np.zeros([NITERS, DIM])
    # t_dist = es.MultivariateGaussian(mu=np.zeros(DIM), cov=np.identity(DIM))
    proposal = es.PCNWalkMove(beta=0.4)
    walker_id = np.random.randint(NWALKERS)
    print 'current walker: %s' % walker_id
    walker = P0[walker_id]
    ensemble = P0[np.concatenate([np.arange(NWALKERS)[:walker_id], np.arange(NWALKERS)[walker_id+1:]])]
    for i in range(NITERS):
        if i % 10000 == 0:
            print "iter: %s" % i
        proposals[i] = proposal.propose(walker, ensemble)[0]

    f_name = './results/proposals.npz'
    print 'saving to %s...' % f_name
    np.savez_compressed('./results/proposals.npz', proposals)

    if DIM == 1:
        plt.hist(proposals)
        plt.savefig('%s proposals with p0=%s' % (NITERS, P0))
    elif DIM == 2:
        plt.hist2d(proposals)
        plt.savefig('%s proposals with p0=%s' % (NITERS, P0))


if __name__ == '__main__':
    main()