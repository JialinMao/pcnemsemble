import numpy as np
from pandas import DataFrame


class Chain(object):
    """
    Chain object, stores history of samples, lnprobs, accepted & extra_data
    """
    def __init__(self, dim=1, nwalkers=1, max_len=1, extra={}):
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

        self.reset()

        self.p = None

    def reset(self):
        """
        Reset history and current position to empty. 
        """
        self._chain = np.empty([self._nwalkers, self._dim, self._max_len], dtype=float)
        self._lnprob = np.empty([self._nwalkers, self._max_len], dtype=float)
        self._accepted = np.empty([self._nwalkers, ], dtype=int)

        self.history = {'chain': self._chain, 'lnprob': self._lnprob, 'accepted': self._accepted}
        self.history.update(self._extra)
        self.p = None

    def update(self, data, name, niter):
        """
        Update chain, lnprob, accepted & extras.
         
        :param data: 
        :param name: 
        :param niter: 
        :return: 
        """

    def get(self, name=None):
        """
        Get `name` history from the `niter`-th iteration
        
        :param name: (optional) the name of the history to get, can be a list. Return all if None.
        
        :return: the history as required.
        """
        idx = self.history.keys() if name is None else name
        return [self.history.get(i) for i in idx]

    @property
    def curr_pos(self):
        """
        :return: current position of all walkers, array of shape [nwalkers, dim] 
        """
        return self.p

    @curr_pos.setter
    def curr_pos(self, p):
        self.p = p

    @property
    def max_len(self):
        return self._max_len

    @max_len.setter
    def max_len(self, N):
        self._max_len = N