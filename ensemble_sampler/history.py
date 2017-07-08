from __future__ import division

import numpy as np
import h5py
import os


class History(object):
    """
    History object, stores history of samples, lnprobs, accepted & extra_data
    """
    def __init__(self, dim=1, nwalkers=1, niter=1, save_every=1, extra={}):
        """
        Initiate a `history` object. 
        Records sample history, lnprob history, acceptance history and extra information 
        for niter iterations. Dim, nwalkers, niter, sample_every can be set later.
        
        :param dim: dimension of the sample space
        :param nwalkers: number of walkers
        :param niter: number of iterations 
        :param save_every: max len of stored history
        :param extra: Dictionary {'blah': dim(blah)} of extra information to store.
        """
        self.dim = dim
        self.nwalkers = nwalkers
        self.niter = niter
        self.curr_pos = None
        self.save_fname = ""

        self._max_len = save_every or niter

        self._name_to_dim = {'chain': dim, 'accepted': 1}
        self._name_to_dim.update(extra)

        self._dict = {k: np.zeros([self.nwalkers, self._max_len, v])
                      for k, v in zip(self._name_to_dim.keys(), self._name_to_dim.values())}

    def add_extra(self, extra={}):
        self._name_to_dim.update(extra)
        self._dict.update({k: np.zeros([self.nwalkers, self._max_len, v])
                           for k, v in zip(extra.keys(), extra.values())})

    def update(self, itr, walker_idx=slice(None), **kwargs):
        """
        Updating the information of _walker_idx_th walker.
        Info is passed in through kwargs in the form name=data
        """
        if itr >= self._max_len:
            itr = itr % self._max_len
        for k in self._dict.keys():
            if kwargs.get(k) is not None:
                self._dict[k][walker_idx, itr, :] = kwargs.get(k)

    def move(self, new_pos, walker_idx=slice(None)):
        self.curr_pos[walker_idx] = new_pos

    def save_to(self, save_dir, title):
        self.save_fname = os.path.join(save_dir, title+'.hdf5')
        print 'saving to ' + self.save_fname + '...'
        if os.path.isfile(self.save_fname):
            f = h5py.File(self.save_fname, 'r+')
            for name in self._name_to_dim.keys():
                dset = f[name]
                dset.resize(dset.shape[1] + self._max_len, axis=1)
                dset[:, -self._max_len:, :] = self._dict.get(name)
        else:
            f = h5py.File(self.save_fname, 'w')
            for name in self._name_to_dim.keys():
                dset = f.create_dataset(name, (self.nwalkers, self._max_len, self._name_to_dim[name]), dtype='f',
                                        maxshape=(self.nwalkers, self.niter, self._name_to_dim[name]), chunks=True)
                dset[...] = self._dict.get(name)
        f.close()

    def get(self, name=None, get_every=1, hdf5=False):
        """
        Get `name` from history, return all if name is None. 
        if hdf5 is True, return h5py dataset
        """
        if os.path.isfile(self.save_fname):
            f = h5py.File(self.save_fname, 'r')
            if not isinstance(name, list):
                dset = f[name] if hdf5 else f[name][::get_every]
            else:
                idx = self._dict.keys() if name is None else name
                try:
                    dset = dict([(i, f[i]) for i in idx]) if hdf5 else dict([(i, f[i][::get_every]) for i in idx])
                except KeyError, err:
                    print 'Supported keys: %s' % str(self._dict.keys())
                    raise err
            return dset
        else:
            if not isinstance(name, list):
                return self._dict.get(name)
            idx = self._dict.keys() if name is None else name
            try:
                return dict([(i, self._dict[i]) for i in idx])
            except KeyError, err:
                print err
                print 'Supported keys: %s' % str(self._dict.keys())

    def get_flat(self, name=None, get_every=1, hdf5=False):
        """
        Get history flattened along the `nwalkers` axis, of shape [niter, dim_of_data]. 
        :param name: (optional) the name of the history to get, can be a list. Return all if None.
        :param get_every: (optional) get every _ iterations
        :param hdf5: return hdf5 file object if True
        :return: Dictionary of inquired history {name: value}.
        """
        if not isinstance(name, list):
            return self.get(name, get_every, hdf5).reshape([-1, self._name_to_dim.get(name)])
        idx = self._dict.keys() if name is None else name
        try:
            return dict([(i, self.get(i, get_every, hdf5).reshape([-1, self._name_to_dim.get(i)])) for i in idx])
        except KeyError, err:
            print err
            print 'Supported keys: %s' % str(self._dict.keys())

    def plot_scatter(self, dim, kind='kde'):
        """
        Scattered plot for two chosen dimensions. Not sure whether this makes sense...
        dim should be list of pairs of integers [[a_1, b_1], [a_2, b_2], ...]
        """
        import seaborn as sns
        from pandas import DataFrame
        for i in dim:
            x, y = ['dim_%s' % int(i[k]+1) for k in range(2)]
            chain = self.get_flat('chain')
            df = DataFrame(np.vstack([chain[:, i[0]], chain[:, i[1]]]).T, columns=[x, y])
            sns.jointplot(x=x, y=y, data=df, kind=kind)

    @property
    def acceptance_rate(self):
        return np.sum(self._dict.get('accepted'), axis=1) / float(self.niter)

    @property
    def history(self):
        return self._dict

    @property
    def names(self):
        return self._dict.keys()

    @property
    def max_len(self):
        return self._max_len

    @max_len.setter
    def max_len(self, N):
        self._max_len = N
        self._dict = {k: np.zeros([self.nwalkers, self._max_len, v])
                      for k, v in zip(self._name_to_dim.keys(), self._name_to_dim.values())}


