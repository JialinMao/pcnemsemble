import numpy as np
from .proposal import Proposal


class WalkMove(Proposal):

    def __init__(self, s):
        """
        Propose with walk move        
        :param s: number of ensemble to use
        """
        assert s >= 2, "Walk move must use an ensemble size larger than 2"
        self.s = s
        super(Proposal).__init__()

    def propose(self, curr_walker, ensemble, random=None):
        """
        
        :param curr_walker: 
            position of the walker(s) we want to move, array of shape (n, dim)
        :param ensemble: 
            ensemble from which we can choose  propose the move, array of shape (m, dim)
        :param random:
            random number generator. Use default if None.
    
        :return: proposed move of shape (Nc, dim)
        """
        rand = np.random.RandomState() if random is None else random
        n, dim = curr_walker.shape[0]
        m = ensemble.shape[0]
        new_pos = np.empty([N, dim])

        for i in range(n):
            idx = rand.choice(np.arange(m), self.s)
            cov = np.cov(ensemble[idx].T)
            new_pos[i] = rand.multivariate_normal(curr_walker, cov)

        return new_pos
