from __future__ import division

from utils import *
import numpy as np
from emcee import autocorr

# NOTE: For plotting with seaborn. Can be commented out if do not have package installed.
import seaborn as sns
from pandas import DataFrame


class History(object):
    """
    History object, stores history of samples, lnprobs, accepted & extra_data
    """
    def __init__(self, dim=1, nwalkers=1, niter=1, extra={}):
        """
        Initiate a chain object. 
        Records sample history, lnprob history, acceptance history and extra information 
        for niter iterations. Dim, nwalkers, niter, sample_every can be set later.
        
        :param dim: dimension of the sample space
        :param nwalkers: number of walkers
        :param niter: number of iterations 
        :param extra: Dictionary {'blah': dim(blah)} of extra information to store.
        """
        self._dim = dim
        self._nwalkers = nwalkers
        self._niter = niter

        self._name_to_dim = {'chain': dim, 'lnprob': 1, 'accepted': 1}
        self._name_to_dim.update(extra)

        self._recording_idx = 0
        self._history = None
        self._curr_pos = None

        self.reset()

    def reset(self):
        """
        Reset history and current position to empty. 
        """
        self._history = {k: np.zeros([self._nwalkers, self._niter, v])
                         for k, v in zip(self._name_to_dim.keys(), self._name_to_dim.values())}
        self._recording_idx = 0
        self._curr_pos = None

    def get(self, name=None):
        """
        Get `name` from history, return all if name is None. 
        """
        if not isinstance(name, list):
            return self._history.get(name)
        idx = self._history.keys() if name is None else name
        try:
            return dict([(i, self._history[i]) for i in idx])
        except KeyError, err:
            print err
            print 'Supported keys: %s' % str(self._history.keys())

    def get_flat(self, name=None):
        """
        Get history flattened along the `nwalkers` axis, of shape [niter, dim_of_data]. 
        :param name: (optional) the name of the history to get, can be a list. Return all if None.
        :return: Dictionary of inquired history {name: value}.
        """
        if not isinstance(name, list):
            return self._history.get(name).reshape([-1, self._name_to_dim.get(name)])
        idx = self._history.keys() if name is None else name
        try:
            return dict([(i, self._history.get(i).reshape([-1, self._name_to_dim.get(i)])) for i in idx])
        except KeyError, err:
            print err
            print 'Supported keys: %s' % str(self._history.keys())

    def get_every(self, get_every, name=None):
        """
        Return (flattened) history taken every `get_every` steps. 
        """
        if not isinstance(name, list):
            return self.get_flat(name)[::get_every]
        idx = self._history.keys() if name is None else name
        try:
            return dict([(i, self.get_flat(name)[i][:, ::get_every]) for i in idx])
        except KeyError, err:
            print err
            print 'Supported keys: %s' % str(self._history.keys())

    def update(self, walker_idx=slice(None), itr=None, **kwargs):
        """
        Updating the information of _walker_idx_th walker.
        Info is passed in through kwargs in the form name=data
        """
        i = itr
        if i is None:
            i = self._recording_idx
            self._recording_idx += 1
        for k in self._history.keys():
            if kwargs.get(k) is not None:
                self._history[k][walker_idx, i, :] = kwargs.get(k)

    def move(self, new_pos, walker_idx=slice(None)):
        self._curr_pos[walker_idx] = new_pos

    def plot_trajectory(self, walker_id, dim, start_from=0):
        """
        Plot the trajectory of selected walker in the selected dimension(s).
        dim should be an integer array.
        """
        assert isinstance(walker_id, (int, float)), "Trajectory plotting is supported for SINGLE WALKER only."
        plot_trajectory(dim, self.get('chain')[walker_id], start_from=start_from)

    def plot_hist(self, dim, walker_id=None, start_from=0):
        """
        Plot histogram of samples in selected dimension(s). 
        Samples from walkers in `walker_id` will be stacked to give the final plot.
        """
        plot_hist(dim, self.get_flat('chain')[walker_id], start_from=start_from)

    def plot_scatter(self, dim):
        """
        Scattered plot for two chosen dimensions. Not sure whether this makes sense...
        dim should be list of pairs of integers [[a_1, b_1], [a_2, b_2], ...]
        """
        for i in dim:
            x, y = ['dim_%s' % int(i[k]+1) for k in range(2)]
            chain = self.get_flat('chain')
            df = DataFrame(np.vstack([chain[:, i[0]], chain[:, i[1]]]).T, columns=[x, y])
            sns.jointplot(x=x, y=y, data=df)

    def auto_corr(self, low=10, high=None, step=1, c=5, fast=False):
        """
        Adopted from emcee.ensemble. See emcee docs for detail. 
        """
        return autocorr.integrated_time(np.mean(self.get("chain"), axis=0), axis=0,
                                        low=low, high=high, step=step, c=c,
                                        fast=fast)

    @property
    def acceptance_rate(self):
        return np.sum(self._history.get('accepted'), axis=1) / float(self._niter)

    @property
    def history(self):
        return self._history

    @property
    def curr_pos(self):
        return self._curr_pos

    @curr_pos.setter
    def curr_pos(self, p):
        self._curr_pos = p

    @property
    def niter(self):
        """
        The length of history. Typically should be number_of_iterations // record_every. 
        """
        return self._niter

    @niter.setter
    def niter(self, N):
        """
        Set the number of iterations, will trigger reset (including the current position).
        """
        self._niter = N
        self.reset()

