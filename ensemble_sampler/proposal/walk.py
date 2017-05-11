import numpy as np
from proposal import Proposal

__all__ = ['PCNWalkMove']


class PCNWalkMove(Proposal):

    def __init__(self, s=None, beta=None, scale=1.0):
        """
        Propose with walk move        
        :param s: number of ensemble to use, if None, do not use ensemble, and use isotropic gaussian with given scale.
        :param beta: hyper-parameter related to sample scale. Should be adjusted to get a good acceptance ratio. 
        """
        if s is not None:
            assert s >= 2, "Walk move must use an ensemble size larger than 2"
            if beta is not None:
                assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        self.s = s
        self.beta = beta
        self.scale = scale
        super(Proposal, self).__init__()

    def propose(self, walkers_to_move, ensemble, ens_idx=None, random=None, *args, **kwargs):
        """
        Give the proposed next position of walkers in `walkers_to_move`
        based on X(t+1) = sqrt(1 - beta ** 2) X(t) + beta * N(0, Cov(ensemble)).
        When beta = 1.0, just the isotropic gaussian random walk proposal.
        
        :param walkers_to_move: 
            position of the walker(s) to move, array of shape (n, dim)
        :param ensemble: 
            ensemble from which we can calculate the covariance matrix for proposal.
            If ens_idx is None, ensemble should be array of shape (m, dim), where each row is an available walker.
            Otherwise ensemble should be array of shape (n + m, dim), with all walkers in the ensemble.
        :param ens_idx:
            Index of available walkers in the ensemble. See above for details.
        :param random:
            random number generator. Use default if None.
    
        :return: proposed move of shape (Nc, dim)
        """
        rand = np.random.RandomState() if random is None else random
        scale = kwargs.get('scale', self.scale)
        beta = kwargs.get('beta', self.beta)
        s = kwargs.get('s', self.s)

        n, dim = walkers_to_move.shape
        m = ensemble.shape[0] if ens_idx is None else len(ens_idx)

        assert s <= m, "Not enough walkers to use %d ensembles" % s

        new_pos = np.zeros_like(walkers_to_move)

        # NOTE: Is it OK if we choose n ensembles at once?
        for i in range(n):
            if self.s is not None:
                available_idx = ens_idx if ens_idx is not None else np.arange(m)
                idx = rand.choice(available_idx, s, replace=False)
                cov = np.cov(ensemble[idx].T)
            else:
                cov = np.identity(dim)
            if beta is not None:
                new_pos[i] = np.sqrt(1 - beta ** 2) * walkers_to_move[i] \
                             + beta * rand.multivariate_normal(np.zeros_like(walkers_to_move[i]), cov)
            else:
                new_pos[i] = rand.multivariate_normal(walkers_to_move[i], scale**2 * cov)

        return new_pos
