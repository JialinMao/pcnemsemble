import numpy as np
from .walk import WalkMove

__all__ = ['PCNMove']


class PCNMove(WalkMove):
    def __init__(self, s, beta):
        self.beta = beta
        super(WalkMove, self).__init__(s=s)

    def propose(self, walkers_to_move, ensemble, ens_idx=None, beta=None, random=None, *args, **kwargs):
        """
        X(t+1) = sqrt(1 - beta ** 2) X(t) + beta * N(0, Cov(ensemble)).
        Others are the same as super class.
        """
        rand = np.random.RandomState() if random is None else random
        beta = self.beta if beta is None else beta

        n, dim = walkers_to_move.shape
        m = ensemble.shape[0] if ens_idx is None else len(ens_idx)

        assert self.s <= m, "Not enough walkers to use %d ensembles" % self.s

        new_pos = np.empty_like(walkers_to_move)

        for i in range(n):
            available_idx = ens_idx or np.arange(m)
            idx = rand.choice(available_idx, self.s)
            cov = np.cov(ensemble[idx].T)

            new_pos[i] = beta * rand.multivariate_normal(np.sqrt(1 - beta ** 2) * walkers_to_move, cov)

        return new_pos

