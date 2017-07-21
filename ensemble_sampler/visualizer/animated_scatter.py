import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import sys; sys.path.insert(0, "../../")

from emcee.autocorr import *
from ensemble_sampler import *

import argparse



class AnimatedScatter(object):
    def __init__(self, history, start=None, prefix=""):
        self.chain = history.get('chain')[:, :start, :]
        self.new_pos = history.get('new_pos')[:, :start, :]
        self.accepted = history.get('accepted')[:, :start, :]

        self.nwalkers, self.niters, self.dim = self.chain.shape
        self.clrs = matplotlib.colors.cnames.keys()[:self.nwalkers]

        self.fig = plt.figure(figsize=(16, 16))
        self.ax = self.fig.add_subplot(111, xlim=(-2, 2), ylim=(-2, 2))
        self.ax.set_title("%s_proposal_history" % prefix)
        self.ani = FuncAnimation(self.fig, self.update, interval=200, init_func=self.init,
                                 frames=self.nwalkers*self.niters, blit=True, repeat=False)

    def init(self):
        pos = self.chain[:, 0, :]
        for i in range(self.nwalkers):
            x, y = pos[i, 0], pos[i, 1]
            self.scat = self.ax.scatter(x, y, c=self.clrs[i], s=100, label='walker_%s' % i, animated=True)
        self.ax.legend()
        return self.scat,

    def update(self, i):
        loc = i // self.nwalkers
        walker_id = i % self.nwalkers
        pos = self.chain[:, loc, :]
        new_pos = self.new_pos[:, loc, :]
        ensemble_id = np.delete(np.arange(self.nwalkers), walker_id)
        self.ax.clear()
        self.ax.set_xlim((-2, 2))
        self.ax.set_ylim((-2, 2))
        for k in range(self.nwalkers):
            if k in ensemble_id:
                self.ax.scatter(pos[k, 0], pos[k, 1],
                                c=self.clrs[k], s=100, label='walker_%s' % k)
            else:
                if not self.accepted[walker_id, loc]:
                    self.ax.scatter(pos[walker_id, 0], pos[walker_id, 1], marker='>',
                                    c=self.clrs[walker_id], s=200, label='walker_%s_(moving)' % walker_id)
                else:
                    self.ax.scatter(new_pos[walker_id, 0], new_pos[walker_id, 1], marker='>',
                                    c=self.clrs[walker_id], s=200, label='walker_%s_(moving)' % walker_id)
                    self.chain[walker_id, loc, :] = new_pos[walker_id]

        if not self.accepted[walker_id, loc]:
            self.ax.scatter(new_pos[walker_id, 0], new_pos[walker_id, 1], marker='x',
                            c='k', s=200, label='proposal')
        self.ax.legend(loc=2)

        return self.scat,

    def save(self, f_name='test.mp4'):
        self.ani.save(f_name)


def plot_fake_proposal(sampler, proposal, t_dist,
                       p0, walker_to_move=0, N=100, prefix=""):
    nwalkers = sampler.nwalkers
    clr = matplotlib.colors.cnames.keys()[:nwalkers]

    fig = plt.figure(figsize=(16, 16))
    ax1 = fig.add_subplot(211, xlim=(-2, 2), ylim=(-2, 2))
    ax2 = fig.add_subplot(212)

    curr_walker = p0[None, walker_to_move, :]
    ensemble = p0[1:, :]
    curr_lnprob = t_dist.get_lnprob(curr_walker)
    accepted_proposals = []
    unaccepted_proposals = []

    for i in range(N):
        p, blobs = proposal.propose(walkers_to_move=curr_walker, ensemble=ensemble)
        proposal_lnprob = t_dist.get_lnprob(p)
        ln_transition_prob_1 = proposal.ln_transition_prob(p, curr_walker)
        ln_transition_prob_2 = proposal.ln_transition_prob(curr_walker, p)
        ln_acc_prob = (proposal_lnprob + ln_transition_prob_1) - (curr_lnprob + ln_transition_prob_2)
        accept = (np.log(np.random.uniform(size=1)) < ln_acc_prob)
        if accept:
            accepted_proposals.append(p)
        else:
            unaccepted_proposals.append(p)
    accepted_proposals = np.array(accepted_proposals).squeeze()
    unaccepted_proposals = np.array(unaccepted_proposals).squeeze()

    for k in range(nwalkers):
        if k == walker_to_move:
            ax1.scatter(p0[k, 0], p0[k, 1], c=clr[k], s=100, marker='>', label='walker_%s_(moving)' % k)
        else:
            ax1.scatter(p0[k, 0], p0[k, 1], c=clr[k], s=100, label='walker_%s' % k)
    ax2.scatter(accepted_proposals[:, 0], accepted_proposals[:, 1], marker='o',
                alpha=0.8, label='accepted_proposals')
    ax2.scatter(unaccepted_proposals[:, 0], unaccepted_proposals[:, 1], marker='x',
                alpha=0.6, label='unaccepted_proposals')

    ax1.legend()
    ax2.legend()
    plt.title(prefix)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--main-1', action='store_true')
    args = args.parse_args()
    dim = 2
    nwalkers = 4
    niters = 5000

    mu = np.zeros(dim)
    cov = np.identity(dim)

    t_dist = MultivariateGaussian(cov=cov, mu=mu, dim=dim)

    p = PCNWalkMove(beta=0.8)
    s = Sampler(t_dist=t_dist, proposal=p, nwalkers=nwalkers)

    p0 = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    try:
        s.run_mcmc(niters, batch_size=1, p0=p0, store=True, store_every=50000, title='test')
    except:
        print "err"

    chain = s.history.get('chain')
    start = np.where(chain[0, :, 0] == 0)[0][0]
    new_pos = s.history.get('new_pos')
    accepted = s.history.get('accepted')
    if args.main_1:
        a = AnimatedScatter(s.history, start, "test")
        a.save()
    else:
        for i in range(4):
            plot_fake_proposal(loc=0, walker_id=i, N=1000, prefix="test")

