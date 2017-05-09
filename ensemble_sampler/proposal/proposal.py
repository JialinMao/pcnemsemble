"""
Abstract class for proposals.
"""


class Proposal(object):

    def __init__(self):
        pass

    def propose(self, curr_walker, ensemble):
        raise NotImplementedError

