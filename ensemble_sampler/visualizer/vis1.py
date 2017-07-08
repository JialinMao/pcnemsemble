from pylab import *
from visualizer import Visualizer

__all__ = ['Visualizer1']


class Visualizer1(Visualizer):

    def __init__(self, history, realtime=False, **kwargs):
        self.history = history
        self.nwalkers = history.nwalkers
        self.N = kwargs.get('max_len', 1)
        self.realtime = realtime

        self.fig = plt.figure(figsize=kwargs.get('figsize', (10, 10)))

        self.x_width = (kwargs.get('xmin', -5), kwargs.get('xmax', 5))
        self.y_width = (kwargs.get('ymin', -5), kwargs.get('ymax', 10))
        self.sample_width = (0, self.N)

        self.ax1 = self.fig.add_subplot(221, xlim=self.x_width, ylim=self.sample_width)
        self.ax2 = self.fig.add_subplot(224, xlim=self.sample_width, ylim=self.y_width)
        self.ax3 = self.fig.add_subplot(223, xlim=self.x_width, ylim=self.y_width, xlabel='x', ylabel='y')

        self.fig.subplots_adjust(wspace=0.0, hspace=0.0)

        self.lines = {}
        cnames = matplotlib.colors.cnames.keys()
        for i in range(self.nwalkers):
            # trajectory of x
            self.lines['line%d1' % i], = self.ax1.plot([], [], cnames[i], linewidth=1)
            # trajectory of y
            self.lines['line%d2' % i], = self.ax2.plot([], [], cnames[i], linewidth=1, label='walker_%d' % i)
            # Current position
            self.lines['line%d3' % i], = self.ax3.plot([], [], 'o', c=cnames[i], linewidth=2, alpha=.1)
            # Line from last position to current position
            self.lines['line%d4' % i], = self.ax3.plot([], [], cnames[i], linewidth=1, alpha=.3)
            # Line to x hist
            self.lines['line%d5' % i], = self.ax3.plot([], [], cnames[i], linewidth=1)
            # Line to y hist
            self.lines['line%d6' % i], = self.ax3.plot([], [], cnames[i], linewidth=1)

        self.ax2.legend()
        self.ax1.set_xticklabels([])
        self.ax2.set_yticklabels([])

        self.i = -1
        self.chain = None
        self.print_every = kwargs.get('print_every', 100)
        super(Visualizer, self).__init__()

    def init(self):
        self.i = -1
        for line in self.lines.values():
            line.set_data([], [])
        return self.lines.values()

    def __call__(self, h):

        if self.i < 0:
            self.i += 1
            if isinstance(h, int):
                self.chain = self.history.get('chain')
            else:
                self.chain = np.empty([self.nwalkers, self.N, 2])
            return self.lines.values()

        if isinstance(h, int):
            pos = self.chain[:, h, :]
        else:
            pos, _, _ = h  # pos.shape = [nwalkers, dim]
            self.chain[:, self.i, :] = pos
        self.i += 1

        if self.i % self.print_every == 0:
            print self.i
        if self.realtime:
            self.ax1.figure.canvas.draw()
            self.ax2.figure.canvas.draw()
            self.ax3.figure.canvas.draw()

        for k in range(self.nwalkers):
            x, y = pos[k]
            self.lines['line%d1' % k].set_data(self.chain[k, :, 0][::-1][-self.i:], range(self.i))
            self.lines['line%d2' % k].set_data(range(self.i), self.chain[k, :, 1][::-1][-self.i:])
            self.lines['line%d3' % k].set_data(self.chain[k, :self.i, 0], self.chain[k, :self.i, 1])
            self.lines['line%d4' % k].set_data(self.chain[k, :self.i, 0], self.chain[k, :self.i, 1])
            self.lines['line%d5' % k].set_data([x, x], [y, self.y_width[1]])
            self.lines['line%d6' % k].set_data([x, self.x_width[1]], [y, y])

        return self.lines.values()
