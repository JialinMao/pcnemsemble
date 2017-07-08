"""
A visualizer for low dimensional target distribution, with not too many walkers.
Similar to http://twiecki.github.io/blog/2014/01/02/visualizing-mcmc/.
"""
from pylab import *

__all__ = ['Visualizer']


class Visualizer(object):
    """
    Abstract class for visualization
    """
    def __init__(self, **kwargs):
        pass

    def init(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()



