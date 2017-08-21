import numpy as np
from proposal import Proposal

__all__ = ['Stretch']


class Stretch(Proposal):

    def __init__(self, a, **kwargs):
        """
        Propose a stretch move.
        :param a:
            parameter that controls the proposal scale.
        """
        self.a = float(a)
        self.z = None
        self.dim = 2
        self.counter = 0
        super(Proposal, self).__init__()

    def propose(self, walkers_to_move, ensemble, random=None, *args, **kwargs):

        rand = np.random.RandomState() if random is None else random
        n, self.dim = walkers_to_move.shape
        m = ensemble.shape[0]

        self.z = ((self.a - 1.0) * rand.uniform(size=[n, 1]) + 1.0) ** 2. / self.a

        c_walkers = ensemble[rand.randint(m, size=(n, ))]
        new_pos = c_walkers + self.z * (walkers_to_move - c_walkers)

        return new_pos, None

    def ln_transition_prob(self, x, y):
        """
        A bit of hack to avoid heavy computation. 
        """
        assert self.z is not None
        if self.counter == 0:
            prob = (self.dim - 1.0) * np.log(self.z.squeeze())
        else:
            prob = 0.0
        self.counter = (self.counter + 1) % 2
        return prob
