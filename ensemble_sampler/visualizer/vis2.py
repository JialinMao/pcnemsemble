import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visualizer import Visualizer

__all__ = ['Visualizer2']


class Visualizer2(Visualizer):
    def __init__(self, history, prefix='', **kwargs):
        self.chain = history.get('chain')
        self.x = history.get('x')
        self.new_pos = history.get('new_pos')
        self.accepted = history.get('accepted')
        self.nwalkers = history.nwalkers

        self.prefix = prefix
        self.c = matplotlib.colors.cnames.keys()[:self.nwalkers]

        self.fig = plt.figure(figsize=kwargs.get('figsize', (16, 16)))
        self.x_width = kwargs.get('x_width', (-2, 2))
        self.y_width = kwargs.get('y_width', (-2, 2))

        self.ax = self.fig.add_subplot(111, xlim=self.x_width, ylim=self.y_width)
        self.scat = self.ax.scatter([], [])
        self.anim = animation.FuncAnimation(self.fig, self.__call__, interval=kwargs.get('interval', 20),
                                            init_func=self.init, blit=True)

        super(Visualizer2, self).__init__()

    def init(self):
        for k in xrange(self.nwalkers):
            self.scat = self.ax.scatter(self.chain[k, 0, 0], self.chain[k, 0, 1], marker='o', c=self.c[k], label='walker_%s' % k)
        return self.scat

    def __call__(self, *args, **kwargs):


