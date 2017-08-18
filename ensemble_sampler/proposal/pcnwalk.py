import numpy as np
from proposal import Proposal

__all__ = ['PCNWalk']


class PCNWalk(Proposal):

    def __init__(self, beta=0.4):
        """
        Generate a Gaussian r.v. W_t ~ N(0, C), where C = cov(ensemble)
        Make a proposal using the strategy:
         X_{t+1} = mu + sqrt(1 - beta ** 2) (X_t - mu) + beta * W_t, where mu = sample mean
        """
        assert 0 <= beta <= 1, "beta must be in [0, 1]"
        self.beta = beta
        self.sample_mean = None
        self.precision = None
        super(Proposal, self).__init__()

    def propose(self, walkers_to_move, ensemble, random=None, *args, **kwargs):
        """
        walkers_to_move.shape = (batch_size, dim)
        ensemble.shape = (Nc, dim)
        """
        rand = np.random.RandomState() if random is None else random
        beta = kwargs.get('beta', self.beta)

        B, _ = walkers_to_move.shape
        Nc, dim = ensemble.shape

        self.sample_mean = np.mean(ensemble, axis=0)
        diff = ensemble - self.sample_mean
        C = 1.0 / (Nc - 1) * np.dot(diff.T, diff)
        self.precision = np.linalg.inv(C)
        W = rand.multivariate_normal(mean=np.zeros(dim), cov=C, size=B)
        proposal = self.sample_mean + np.sqrt(1 - beta ** 2) * (walkers_to_move - self.sample_mean) + beta * W

        return proposal, None

    def ln_transition_prob(self, x, y):
        """
        Calculate ln transition probability from x -> y
        :param x: start position, shape=(batch_size, dim)
        :param y: end position, shape=(batch_size, dim)
        :return: prob, shape=(batch_size, 1), extra info for debugging
        """
        mu = self.sample_mean + np.sqrt(1 - self.beta ** 2) * (x - self.sample_mean)
        diff = (y - mu) / self.beta
        return - 0.5 * np.einsum('ij, ji->i', diff, np.dot(self.precision, diff.T))
