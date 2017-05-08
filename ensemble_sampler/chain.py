import numpy as np
from pandas import DataFrame


class Chain(object):
    """
    Chain object, stores history of samples, lnprobs, accepted & extra_data
    """
    def __init__(self, dim=1, nwalkers=1, max_len=1, extra=[]):
        """
        Initiate a chain object. 
        Records sample history, lnprob history, acceptance history and extra information 
        for max_len iterations. Dim, nwalkers, niter can be set later.
        
        :param dim: dimension of the sample space
        :param nwalkers: number of walkers
        :param max_len: length of history
        :param extra: extra information to record
        """
        self.dim = dim
        self.nwalkers = nwalkers
        self.max_len = max_len
        self.index = ['sample', 'lnprob', 'accepted'] + extra

        self.chain = np.empty([max_len, dim, nwalkers, len(self.index)])
        self.p = np.empty([dim, nwalkers])

    def reset(self):
        """
        Reset history and current position to empty. 
        """
        self.chain = np.empty([self.max_len, self.dim, self.nwalkers, len(self.index)])
        self.p = np.empty([self.dim, self.nwalkers])

    def update(self, niter, name, data):
        """
        Update history. 
        
        :param niter: python slice of position to perform update. 
        :param name: a list of index of updates
        :param data: the data to write, with shape [n, len(name), dim, nwalkers]
        """
        self.chain[niter, :, :, np.hstack([self.index.index(i) for i in name])] = data

    def get(self, name, niter=slice(None)):
        return self.chain[niter, :, :, np.hstack([self.index.index(i) for i in name])]

    @property
    def curr_pos(self):
        return self.p

    @curr_pos.setter
    def curr_pos(self, p):
        self.p = p


    @property
    def dim(self):
        return self.dim

    @dim.setter
    def dim(self, dim):
        self.dim = dim

    @property
    def nwalkers(self):
        return self.nwalkers

    @nwalkers.setter
    def nwalkers(self, n):
        self.nwalkers = n


