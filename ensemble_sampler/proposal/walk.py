import numpy as np
from proposal import Proposal

__all__ = ['PCNWalkMove']


class PCNWalkMove(Proposal):

    def __init__(self, s, beta=1.0):
        """
        Propose with walk move        
        :param s: number of ensemble to use
        :param beta: hyper-parameter related to sample scale. Should be adjusted to get a good acceptance ratio. 
        """
        assert s >= 2, "Walk move must use an ensemble size larger than 2"
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        self.s = s
        self.beta = beta
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
        beta = kwargs.get('beta', self.beta)
        s = kwargs.get('s', self.s)

        n, dim = walkers_to_move.shape
        m = ensemble.shape[0] if ens_idx is None else len(ens_idx)

        assert s <= m, "Not enough walkers to use %d ensembles" % s

        new_pos = np.zeros_like(walkers_to_move)

        # NOTE: Is it OK if we choose n ensembles at once?
        for i in range(n):
            # NOTE: Do we need to set replace = False?
            available_idx = ens_idx if ens_idx is not None else np.arange(m)
            idx = rand.choice(available_idx, s)
            cov = np.cov(ensemble[idx].T)
            new_pos[i] = beta * rand.multivariate_normal(np.sqrt(1 - beta ** 2) * walkers_to_move[i], cov)

        return new_pos
