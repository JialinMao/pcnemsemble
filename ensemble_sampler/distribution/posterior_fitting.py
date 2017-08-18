import numpy as np
from distribution import Distribution


class LogPosterior(Distribution):
    def __init__(self, log_prior, t, Y, dim):
        self.dim = dim
        self.m = (dim - 1) / 2
        self.t = t
        self.Y = Y
        self.log_prior = log_prior
        super(LogPosterior, self).__init__(log_prior=log_prior, t=t, Y=Y, dim=dim)

    def get_lnprob(self, p, **kwargs):
        log_probs = np.zeros(len(p))
        for i in range(len(p)):
            A = p[i, :self.m]
            l = p[i, self.m:-1]
            s = p[i, -1]

            x = self.Y - np.dot(np.exp(-np.outer(self.t, l)), A)
            log_prob = - x ** 2 / (2 * s ** 2) - np.log(max(abs(s), 1e-8))
            log_prob = np.sum(log_prob)
            if self.log_prior is not None:
                log_prob += self.log_prior(A, l, s)
            log_probs[i] = log_prob
        return log_probs


class FlatPrior(object):
    def __init__(self, A_max, l_max, s_max):
        self.A_max = A_max
        self.l_max = l_max
        self.s_max = s_max

    def __call__(self, A, l, s):
        val = -(len(A) * np.log(self.A_max * 2) + len(l) * np.log(self.l_max * 2) + np.log(self.s_max * 2))
        if np.all(abs(A) <= self.A_max) and np.all(abs(l) <= self.l_max) and np.all(0 < s <= self.s_max):
            return val
        else:
            return -np.inf


class GaussianPrior(object):
    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov

    def __call__(self, A, l, s):
        """
        Gaussian prior with large variance
        """
        x = np.hstack([A, l, s])
        mu = np.zeros(len(x)) if self.mean is None else self.mean
        c = 10 * np.identity(len(x)) if self.cov is None else self.cov
        inverse_c = np.linalg.inv(c)
        log_prob = np.dot(x - mu, np.dot(inverse_c, x - mu)) + np.log(np.linalg.det(c))
        return - log_prob / 2.0


def generate_fake_data(A, l, s, t):
    """
    :param A, l, s: parameters, s is the nuisance parameter
    :param t: time steps
    :return: t, Y : Y[i] = sum_{j}(A[j] * exp(-l[j] * t[i])) + x_i
    """
    n = len(t)
    f_t = np.dot(np.exp(-np.outer(t, l)), A)
    xi = np.random.normal(loc=0.0, scale=s, size=n)
    Y = f_t + xi
    return {'t': t, 'Y': Y}
