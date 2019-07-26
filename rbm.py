from __future__ import division
from util import *


class RBM(object):
    """
    Bernoulli restricted Boltzmann machine (RBM).
    """
    def __init__(self, m, k, W, bvec, cvec, dropout_units=list()):
        """
        Args:
            m: int, number of visible units.
            k: int, number of hidden units.
            mu: array(d), node potentials.
        """
        assert_shape(W, (m, k))
        assert_shape(bvec, (m,))
        assert_shape(cvec, (k,))

        self.m = m
        self.k = k
        self.W = deepcopy(W)
        self.bvec = deepcopy(bvec)
        self.cvec = deepcopy(cvec)
        self.domain = [0, 1]

        assert set(dropout_units).issubset(range(self.k))
        self.dropout_units = np.array(dropout_units)

        return

    def __repr__(self):
        return "RBM(m=%d, k=%d, W = %s, bvec = %s, cvec = %s, domain = %s)" % \
            (self.m, self.k, self.W, self.bvec, self.cvec, self.domain)

    def __str__(self):
        res = "------------------------------\n"
        res += "RBM(m, k, W, bvec, cvec)\n"
        res += "m = %d\n" % self.m
        res += "k = %d\n" % self.k
        res += "W = %s\n" % np.round(self.W, 3)
        res += "bvec = %s\n" % np.round(self.bvec, 3)
        res += "cvec = %s\n" % np.round(self.cvec, 3)
        res += "domain = %s\n" % self.domain
        res += "------------------------------\n"

        return res

    def check_valid_input(self, x, _type="visible"):
        """
        Check whether all elements of the input is in the discrete domain of
            the model.

        Args:
            x: list/array of arbitrary dimensions.
        """
        x = np.atleast_2d(x)

        if _type == "visible":
            dim = self.m
        elif _type == "hidden":
            dim = self.k
        else:
            raise ValueError("_type = %s not recognized!" % _type)

        assert x.shape[1] == dim, x.shape  # Check dimension

        assert np.all(np.isin(x, self.domain))  # Check values

        return True

    def neg(self, x, i):
        """
        Negate the i-th coordinate of x.
        """
        self.check_valid_input(x)
        x = np.atleast_2d(x)
        assert i < x.shape[1]

        res = deepcopy(x)
        res[:, i] = 1 - res[:, i]  # Flips between 0 and 1

        self.check_valid_input(res)

        return res

    def sample_hidden(self, vvec):
        """
        Given configurations of visible units, sample the hidden units.

        Args:
            vvec: array((num_samples, self.m)), values of the observed units.

        Returns:
            hvec: array((num_samples, self.k)), sampled hidden units.
        """
        self.check_valid_input(vvec, "visible")
        vvec = np.atleast_2d(vvec)
        n, m = vvec.shape  # num_samples, self.m

        cterm = np.tile(self.cvec, (n, 1))  # (n, k)
        wterm = vvec.dot(self.W)  # (n, k)
        probs = sigmoid(cterm + wterm)  # (n, k)
        assert_shape(probs, (n, self.k))

        hvec = rand.binomial(n=1, p=probs)  # (n, k)
        self.check_valid_input(hvec, "hidden")

        return hvec

    def sample_visible(self, hvec):

        """
        Given configurations of visible units, sample the hidden units.

        Args:
            vvec: array((num_samples, self.m)), values of the observed units.

        Returns:
            hvec: array((num_samples, self.k)), sampled hidden units.
        """
        self.check_valid_input(hvec, "hidden")
        hvec = np.atleast_2d(hvec)
        n, k = hvec.shape  # num_samples, self.m

        bterm = np.tile(self.bvec, (n, 1))  # (n, m)
        wterm = hvec.dot(self.W.T)  # (n, m)
        probs = sigmoid(bterm + wterm)  # (n, m)
        assert_shape(probs, (n, self.m))

        vvec = rand.binomial(n=1, p=probs)  # (n, m)
        self.check_valid_input(vvec, "visible")

        return vvec

    def sample(self, num_iters, num_samples=1, burnin_prop=.5, _plot=False):
        """
        Sample hidden and observed units from an RBM via Gibbs sampling.

        Returns:

        """
        num_iters = int(num_iters)
        # v_hist = np.zeros((num_iters, num_samples, self.m))
        # h_hist = np.zeros((num_iters, num_samples, self.k))

        vvec = rand.binomial(n=1, p=.5, size=(num_samples, self.m))
        hvec = rand.binomial(n=1, p=.5, size=(num_samples, self.k))
        self.check_valid_input(vvec, "visible")
        self.check_valid_input(hvec, "hidden")

        # Gibbs sampling
        for t in xrange(num_iters):
            hvec = self.sample_hidden(vvec)
            if self.dropout_units.size > 0:
                hvec[:, self.dropout_units] = 0.

            vvec = self.sample_visible(hvec)
            # h_hist[t, :, :] = hvec
            # v_hist[t, :, :] = vvec

        # # Burn-in
        # start = int(num_iters * burnin_prop) + 1
        # h_hist = h_hist[start::, :]
        # v_hist = v_hist[start::, :]

        # return vvec, hvec, v_hist, h_hist
        return vvec, hvec

    def score(self, vvec):
        """
        Computes the (difference) score function.

        Args:
            x: array((num_samples, m))

        Returns:
            res: array((num_samples, m))
        """
        # NOTE: Heavy use of numpy broadcasting in the computations...

        vvec = np.atleast_2d(vvec)  # (n, m)
        self.check_valid_input(vvec, "visible")
        n, m = vvec.shape
        k = self.k

        means = np.tile(self.cvec, (n, m, 1))
        means += vvec.dot(self.W)[:, np.newaxis, :]
        assert_shape(means, (n, m, k))

        means_neg = means + (1. - 2*vvec[..., np.newaxis]) * self.W  # \neg v_j - v_j
        assert_shape(means_neg, (n, m, k))

        def logsumexp_helper(y):
            """
            For each element y[i, j], compute logsumexp([0., y[i, j]]).
            """
            y_new = y[..., np.newaxis]
            y_zero = np.stack([y_new, np.zeros_like(y_new)], axis=-1)
            res = logsumexp(y_zero, axis=-1).reshape(y.shape)
            return res

        log_term = logsumexp_helper(means_neg) - logsumexp_helper(means)
        assert_shape(log_term, (n, m, k))

        log_temp = (1 - 2*vvec) * self.bvec + np.sum(log_term, axis=-1)
        assert_shape(log_temp, (n, m))

        res = 1. - np.exp(log_temp)

        return res
