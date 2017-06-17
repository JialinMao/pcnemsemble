import numpy as np
from proposal import Proposal

__all__ = ['WalkMove']


class WalkMove(Proposal):

    def __init__(self, ensemble=False, s=None, beta=None, scale=1.0):
        """
        Generate a Gaussian r.v. W ~ N(0, C), 
         where C = cov(s (all if s is None) walkers from ensemble)        , if ensemble = True
         and   C = identity                                               , otherwise.
        Make a proposal using the strategy:
         X_{t+1} = X_t + scale * W_t                                      , if beta is None
         X_{t+1} = mu + sqrt(1 - beta ** 2) (X_t - mu) + beta * W_t
          where mu = sample mean                                          , otherwise. 
        """
        if ensemble and s is not None:
            assert s >= 2, "ensemble size %d too small, must be >= 2" % s
        if beta is not None:
            assert 0 <= beta <= 1, "beta must be in [0, 1]"
        self.s = s
        self.ensemble = ensemble
        self.beta = beta
        self.scale = scale
        super(Proposal, self).__init__()

    def propose(self, walkers_to_move, ensemble, random=None, *args, **kwargs):
        """
        :param walkers_to_move: 
            position of the walker(s) to move, shape = (batch_size, dim)
        :param ensemble: 
            ensemble from which we can calculate the covariance matrix for proposal.
            Should be array of shape (Nc, dim), where each row is an available walker.
        :param random:
            random number generator. Use default if None.
    
        :return: proposed move of shape (Nc, dim)
        """
        rand = np.random.RandomState() if random is None else random

        scale = kwargs.get('scale', self.scale)
        beta = kwargs.get('beta', self.beta)
        s = kwargs.get('s', self.s)

        batch_size, dim = walkers_to_move.shape
        Nc, _ = ensemble.shape

        assert s <= Nc, "%d walkers in ensemble, not enough for %d ensembles" % (Nc, s)

        if not self.ensemble:
            # use isotropic Gaussian proposal.
            proposal = rand.normal(size=[batch_size, dim])
            self.sample_mean = 0
        else:
            if s is None:
                s = Nc
            # use first `s` walkers in the ensemble and propose a gaussian with the same cov.
            idx = np.arange(s)
            self.sample_mean = np.mean(ensemble[idx], axis=0)
            self.C = 1.0 / (Nc - 1) * np.dot((ensemble[idx] - self.sample_mean).T, ensemble[idx] - self.sample_mean)
            proposal = rand.multivariate_normal(mean=np.zeros(dim), cov=self.C, size=batch_size)

        if beta is not None:
            # use pCN proposal
            new_pos = self.sample_mean + np.sqrt(1 - beta ** 2) * (walkers_to_move - self.sample_mean) + beta * proposal
        else:
            # use regular random walk proposal
            new_pos = walkers_to_move + scale * proposal

        return new_pos

    def ln_transition_prob(self, x, y):
        """
        Calculate ln transition probability from x -> y
        :param x: start position, shape=(batch_size, dim) 
        :param y: end position, shape=(batch_size, dim) 
        :return: prob, shape=(batch_size, 1) 
        """
        if self.beta is None:
            return 0.0
        diff = y - np.sqrt(1 - self.beta ** 2) * (x - self.sample_mean) - self.sample_mean
        precision = np.linalg.inv(self.C)
        return - np.dot(diff, np.dot(precision, diff.T)).squeeze() / 2.0
