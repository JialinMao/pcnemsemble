import numpy as np
from pandas import DataFrame


class History(object):
    """
    History object, stores history of samples, lnprobs, accepted & extra_data
    """
    def __init__(self, dim=1, nwalkers=1, max_len=1, extra={}):
        # NOTE: `extra` is not useful at all right now. Further modification needed.
        """
        Initiate a chain object. 
        Records sample history, lnprob history, acceptance history and extra information 
        for max_len iterations. Dim, nwalkers, niter can be set later.
        
        :param dim: dimension of the sample space
        :param nwalkers: number of walkers
        :param max_len: length of history
        :param extra: extra information to record
        """
        self._dim = dim
        self._nwalkers = nwalkers
        self._max_len = max_len
        self._extra = extra

        self._chain, self._lnprob, self._accepted, self._history, self.p = [None] * 5

        self._recording_idx = 0

        self.reset()

    def reset(self):
        """
        Reset history and current position to empty. 
        """
        # NOTE: _chain has length self._max_len // self._nwalkers because we are moving
        # one walker at a time. May need to refactor after parallelization.
        self._chain = np.empty([self._nwalkers, self._dim, self._max_len // self._nwalkers], dtype=float)
        self._lnprob = np.empty([self._nwalkers, self._max_len], dtype=float)
        self._accepted = np.empty([self._nwalkers, ], dtype=int)
        self._recording_idx = 0

        self._update_history()

        self.p = None

    def get(self, name=None):
        """
        Get `name` history from the `niter`-th iteration
        
        :param name: (optional) the name of the history to get, can be a list. Return all if None.
        
        :return: the history as required.
        """
        idx = self.history.keys() if name is None else name
        return [self.history.get(i) for i in idx]

    def update(self, walker_idx, accepted=False, lnprob=None):
        if accepted:
            self._accept(walker_idx)
        self._update_chain(walker_idx)
        self._update_lnprob(walker_idx, lnprob)
        self._recording_idx += 1

    def _accept(self, walker_idx):
        """
        Record an acceptance of walker i. 
        """
        self._accepted[walker_idx] += 1

    def _update_chain(self, walker_idx):
        self._chain[walker_idx, :, self._recording_idx // self._nwalkers] = self.p[walker_idx]

    def _update_lnprob(self, walker_id, lnprob):
        self._lnprob[walker_id, self._recording_idx] = lnprob

    def _update_history(self, extra=None):
        """
        Update chain, lnprob, accepted & extras.
         
        :param extra: update extra if not None
        """
        extra = self._extra if extra is None else extra
        self._history = {'chain': self._chain, 'lnprob': self._lnprob, 'accepted': self._accepted}
        self._history.update(extra)

    @property
    def curr_pos(self):
        """
        :return: current position of all walkers, array of shape [nwalkers, dim] 
        """
        return self.p

    @curr_pos.setter
    def curr_pos(self, p):
        """
        Set current position to p. 
        """
        self.p = p

    @property
    def max_len(self):
        """
        The length of history. Typically should be number_of_iterations // record_every. 
        """
        return self._max_len

    @max_len.setter
    def max_len(self, N):
        self._max_len = N

    @property
    def history(self):
        self._update_history()
        return self._history

