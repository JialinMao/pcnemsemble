import numpy as np
from proposal import Proposal

__all__ = ['PCNWalkMove']


class PCNWalkMove(Proposal):

    def __init__(self, s=None, beta=None, scale=1.0):
        """
        Propose generalized ensemble walk move.
        Use covariance matrix calculated from ensemble if `s` is not None, otherwise use identity matrix.
        Use the strategy X(t+1) = sqrt(1 - beta ** 2) X(t) + beta * N(0, Cov) if `beta` is not None,
        otherwise use simple random walk proposal with scale `scale`.
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

        # new_pos = np.zeros_like(walkers_to_move)

        available_idx = ens_idx if ens_idx is not None else np.arange(m)
        # NOTE: probably should use replace=False. Probably that does not matter, not sure.
        idx = rand.choice(available_idx, [n, s]) if s is not None else None

        if self.s is not None:
            x = ensemble[idx] - np.mean(ensemble[idx], axis=1)[:, None, :]
            w = rand.normal(size=[n, 1, s])
            proposal = np.einsum("ijk, ikl -> ijl", w, x).squeeze()
        else:
            proposal = rand.normal(size=[n, dim])
        if beta is not None:
            new_pos = np.sqrt(1 - beta ** 2) * walkers_to_move + beta * proposal
        else:
            new_pos = walkers_to_move + scale * proposal
        """
        for i in range(n):
            if self.s is not None:
                x = ensemble[idx[i]] - np.mean(ensemble[idx[i]], axis=0)
                z = rand.normal(size=s)
                proposal = np.dot(z, x)
            else:
                # NOTE: shall we use scale here or later?
                proposal = rand.normal()
            if beta is not None:
                new_pos[i] = np.sqrt(1 - beta ** 2) * walkers_to_move[i] + beta * proposal
            else:
                new_pos[i] = walkers_to_move[i] + scale * proposal
        """
        return new_pos
