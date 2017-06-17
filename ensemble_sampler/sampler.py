import os
import numpy as np
from emcee.autocorr import *

from history import History
from visualizer import Visualizer

__all__ = ["Sampler"]


class Sampler(object):

    def __init__(self, t_dist, proposal, nwalkers=1):
        """
        :param t_dist: target distribution
        :param proposal: proposal method
        :param nwalkers: number of walkers to use for sampling, default set to 1 for non-ensemble methods.
        """
        self.dim = t_dist.dim
        self.nwalkers = nwalkers
        self.t_dist = t_dist
        self.proposal = proposal
        self._history = History(dim=self.dim, nwalkers=nwalkers)
        self._acceptances = np.zeros([self.nwalkers, ])
        self._random = np.random.RandomState()

    def init(self):
        """
        Initialize storage in self._history. 
        """
        self._history.init()

    def clear(self):
        """
        Maintain shape of stored history, reset all to 0.
        """
        self._history.clear()

    def sample(self, niter, batch_size, verbose=False, print_every=1000, store=False,
               store_every=None, save_dir='./result/', title=None, **kwargs):

        if batch_size is None:
            batch_size = self.nwalkers // 2
        assert self.nwalkers % batch_size == 0, 'Batch size must divide number of walkers.'

        try:
            os.remove(os.path.join(save_dir, title + '.hdf5'))
        except (OSError, AttributeError):
            pass

        all_walkers = self._history.curr_pos  # (nwalkers, dim)
        ln_probs = self.t_dist.get_lnprob(all_walkers)  # (nwalkers, )

        n = self.nwalkers // batch_size
        debug = False
        for i in xrange(niter):
            if i >= niter / 2 and self.debug:
                import ipdb
                ipdb.set_trace()
                debug = True

            for k in range(n):
                idx = slice(k * batch_size, (k+1) * batch_size)

                curr_walker = all_walkers[idx]  # (batch_size, dim)
                curr_lnprob = ln_probs[idx]  # (batch_size, dim)

                ens_idx = np.zeros(self.nwalkers)
                ens_idx[idx] = -1
                ensemble = all_walkers[ens_idx >= 0]  # (Nc = nwalkers-batch_size, dim)

                proposal = self.proposal.propose(curr_walker, ensemble, self._random, debug=debug, **kwargs)  # (batch_size, dim)

                proposal_lnprob = self.t_dist.get_lnprob(proposal)
                ln_transition_prob_1 = self.proposal.ln_transition_prob(proposal, curr_walker)
                ln_transition_prob_2 = self.proposal.ln_transition_prob(curr_walker, proposal)
                ln_acc_prob = (proposal_lnprob + ln_transition_prob_1) - (curr_lnprob + ln_transition_prob_2)  # (batch_size, )

                accept = (np.log(self._random.uniform(size=batch_size)) < ln_acc_prob)  # (batch_size, )

                self._acceptances[idx] += accept
                ln_probs[idx][accept] = proposal_lnprob[accept]

                self._history.move(walker_idx=np.arange(idx.start, idx.stop)[accept], new_pos=proposal[accept])
                all_walkers = self._history.curr_pos

            if verbose and i % print_every == 0:
                print '====iter %s====' % i
                print 'Acceptance probability: ' + str(np.exp(ln_acc_prob))
                print 'Accepted: ' + str(accept)
                # print 'All walkers: ' + str(all_walkers)
                # print 'ln_probs: ' + str(ln_probs)

            if store:
                if title is None:
                    from datetime import datetime
                    title = datetime.now().strftime('%Y-%m-%d_%H:%M')
                itr = i if store_every is None else i % store_every
                self._history.update(itr=itr, chain=self._history.curr_pos)
                if store_every is not None and i % store_every == 0:
                    self._history.save_to(save_dir, title)
                    self._history.clear()

            elif not kwargs.get('per_walker', False):
                yield self._history.curr_pos, ln_probs, accept

    def run_mcmc(self, niter, **kwargs):

        for h in self.sample(niter, **kwargs):
            pass

    def animation(self, niter, save=False, save_name='animation.mp4', realtime=False, **kwargs):
        """
        :param niter: number of iterations to run 
        :param save: whether to save video 
        :param save_name: the name of saved video 
        :param realtime: whether to render in real time.
        :param kwargs: arguments to pass to self.sample function. 
        :return: matplotlib.animation.FuncAnimation object. Can be converted to HTML5 video tag through
                 animation.to_html5_video
        """
        from matplotlib.animation import FuncAnimation
        hist = self.sample(niter, **kwargs)
        visualizer = Visualizer(self.history, realtime, **kwargs)
        animation = FuncAnimation(fig=visualizer.fig, func=visualizer, init_func=visualizer.init,
                                  frames=hist, interval=20, blit=True, save_count=self.history.max_len)
        if save:
            animation.save(save_name)
        elif realtime:
            visualizer.fig.show()
        else:
            return animation

    def auto_corr(self, low=10, high=None, step=1, c=5, fast=False):
        """
        Adopted from emcee.ensemble. See emcee docs for detail. 
        """
        chain = self.t_dist.get_auto_corr_f(np.mean(self._history.get('chain'), axis=0))
        return integrated_time(chain, axis=0, low=low, high=high, step=step, c=c, fast=fast)

    @property
    def history(self):
        return self._history

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        self._random.set_state(state)

    @property
    def acceptance(self):
        return self._acceptances
