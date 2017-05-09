import numpy as np
from .proposal import Proposal

__all__ = ['WalkMove']


class WalkMove(Proposal):

    def __init__(self, s):
        """
        Propose with walk move        
        :param s: number of ensemble to use
        """
        assert s >= 2, "Walk move must use an ensemble size larger than 2"
        self.s = s
        super(Proposal, self).__init__()

    def propose(self, walkers_to_move, ensemble, ens_idx=None, random=None, *args, **kwargs):
        """
        Give the proposed next position of walkers in `walkers_to_move`
        
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

        n, dim = walkers_to_move.shape
        m = ensemble.shape[0] if ens_idx is None else len(ens_idx)

        assert self.s <= m, "Not enough walkers to use %d ensembles" % self.s

        new_pos = np.empty_like(walkers_to_move)

        # NOTE: Is it OK if we choose n ensembles at once?
        for i in range(n):
            # NOTE: Do we need to set replace = False?
            available_idx = ens_idx or np.arange(m)
            idx = rand.choice(available_idx, self.s)
            cov = np.cov(ensemble[idx].T)
            new_pos[i] = rand.multivariate_normal(walkers_to_move, cov)

        return new_pos
