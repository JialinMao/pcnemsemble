"""
Abstract class for proposals.
"""


class Proposal(object):

    def __init__(self, *args, **kwargs):
        pass

    def propose(self, curr_walker, ensemble):
        raise NotImplementedError("proposal method must be implemented"
                                  "by subclass")

    def ln_transition_prob(self, x, y):
        """
        Log transition probability from x -> y. Return constant 1.0 if the proposal is symmetric.
        """
        return 1.0
