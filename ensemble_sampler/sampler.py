import os
import numpy as np
from emcee.autocorr import *

from history import History
from utils import *

__all__ = ["Sampler"]


class Sampler(object):

    def __init__(self, t_dist, proposal, nwalkers=1, visualizer=None):
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
        self._random = np.random.RandomState()
        self._visualizer = visualizer

    def sample(self, niter, batch_size=1, p0=None, rstate0=None,
               store=False, store_every=None, save_dir='./results', title='', **kwargs):
        """
        Sampling function, yields position, ln_prob and accepted or not every iteration, 
        or saves to hdf5 file every `store_every` iterations if store=True.

        :param niter:
            The number of steps to run.
        :param batch_size:
            In each iteration move `batch_size` walkers simultaneously using all other walkers as ensemble.
        :param p0:
            The initial position vector.  Can also be None to resume from where :func:``run_mcmc`` left off 
            the last time it executed.
        :param rstate0:
            The initial random state. Use default if None.
        :param store:
            Store chain if True, otherwise discard chain and yield results every iteration.
        :param store_every:
            How often to save chain to file and free-up memory. Store the entire chain if None.
        :param save_dir:
            The directory to save history. 
        :param title:
            Title of the saved file.
        :param kwargs:
            Optional keywords arguments for proposal / calculating lnprob.
        """
        if rstate0 is not None:
            self._random.set_state(rstate0)

        assert self.nwalkers % batch_size == 0, 'Batch size must divide number of walkers.'

        self._history.niter = niter
        self._history.max_len = niter if store_every is None else store_every

        if store:
            # Remove file if already exist
            remove_f(title, save_dir)

        if self._history.curr_pos is None:
            if kwargs.get('random_start'):
                p0 = np.random.randn(self.nwalkers * self.dim).reshape([self.nwalkers, -1])
            if p0 is None:
                raise ValueError("Cannot have p0=None if run_mcmc has never been called. "
                                 "Set `random_start=True` in kwargs to use random start")
            else:
                self._history.curr_pos = p0

        all_walkers = self._history.curr_pos  # (nwalkers, dim)
        ln_probs = self.t_dist.get_lnprob(all_walkers)  # (nwalkers, )

        n = self.nwalkers // batch_size
        for i in xrange(niter):
            if store:
                self._history.update(itr=i, chain=self._history.curr_pos)
                if store_every is not None and (i+1) % store_every == 0:
                    self._history.save_to(save_dir, title)

            for k in xrange(n):
                # pick walkers to move
                idx = slice(k * batch_size, (k+1) * batch_size)
                curr_walker = all_walkers[idx]  # (batch_size, dim)
                ens_idx = np.zeros(self.nwalkers)
                ens_idx[idx] = -1
                ensemble = all_walkers[ens_idx >= 0]  # (Nc, dim)

                # propose a move
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        proposal, blob = self.proposal.propose(curr_walker, ensemble, self._random, **kwargs)
                        if i == 0 and k == 0 and blob is not None and store:
                            name_to_dim = dict([(key, blob[key].shape[1]) for key in blob.keys()])
                            self.history.add_extra(name_to_dim)
                        if store:
                            self.history.update(i, idx, **blob)
                    except Warning as e:
                        print 'iteration: ', i
                        print 'walker: ', idx
                        print 'error found:', e

                # calculate acceptance probability
                curr_lnprob = ln_probs[idx]  # (batch_size, )
                proposal_lnprob = self.t_dist.get_lnprob(proposal)
                ln_transition_prob_1 = self.proposal.ln_transition_prob(proposal, curr_walker)
                ln_transition_prob_2 = self.proposal.ln_transition_prob(curr_walker, proposal)
                ln_acc_prob = (proposal_lnprob + ln_transition_prob_1) - (curr_lnprob + ln_transition_prob_2)

                # accept or reject
                accept = (np.log(self._random.uniform(size=batch_size)) < ln_acc_prob)
                if store:
                    self._history.update(i, idx, accepted=accept)
                else:
                    yield self._history.curr_pos, proposal, accept

                # update history records for next iteration
                self._history.move(walker_idx=np.arange(idx.start, idx.stop)[accept], new_pos=proposal[accept])
                all_walkers = self._history.curr_pos
                ln_probs[idx][accept] = proposal_lnprob[accept]

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
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        hist = self.sample(niter, **kwargs)
        visualizer = self._visualizer(self.history, realtime, **kwargs)
        animation = FuncAnimation(fig=visualizer.fig, func=visualizer, init_func=visualizer.init,
                                  frames=hist, interval=20, blit=True, save_count=self.history.max_len, **kwargs)
        if save:
            animation.save(save_name)
        elif realtime:
            plt.show()
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

    @property
    def accepted(self):
        return self._accepted

    @property
    def visualizer(self):
        return self._visualizer

    @visualizer.setter
    def visualizer(self, vis):
        self._visualizer = vis
