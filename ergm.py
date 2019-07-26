from __future__ import division
from util import *


class ERGM(object):
    """
    Exponential random graph model (ERGM).
    """
    def __init__(self, d, rho, theta, tau):
        """
        Args:
            d: int, number of nodes.
            rho: float, param for number of edges.
            theta: float, param for number of 2-stars.
            tau: float, param for number of triangles.
        """
        self.d = d
        self.rho = rho
        self.theta = theta
        self.tau = tau
        self.params = np.array([self.rho, self.theta, self.tau])
        self.domain = [0, 1]  # Binary edges

        return

    def __repr__(self):
        return "ERGM(d=%d, rho = %s, theta = %s, tau = %s, domain=%s)" % \
            (self.d, self.rho, self.theta, self.tau, self.domain)

    def __str__(self):
        res = "------------------------------\n"
        res += "ERGM(d, rho, theta, tau, domain)\n"
        res += "d = %d\n" % self.d
        res += "rho = %s\n" % np.round(self.rho, 3)
        res += "theta = %s\n" % np.round(self.theta, 3)
        res += "domain = %s\n" % self.domain
        res += "------------------------------\n"

        return res

    def check_valid_input(self, x):
        """
        Check whether all elements of the input is in the discrete domain of
            the model.

        Args:
            x: list/array of arbitrary dimensions.
        """
        x = np.atleast_2d(x)
        assert 2*x.shape[1] == self.d*(self.d-1)

        assert np.all(np.isin(x, self.domain))  # Check values

        return True

    def neg(self, x, i):
        """
        Flip the $i$-th edge
        """
        self.check_valid_input(x)
        x = np.atleast_2d(x)
        assert i < x.shape[1]

        res = deepcopy(x)
        res[:, i] = 1 - res[:, i]  # Flips between 0 and 1

        self.check_valid_input(res)

        return res

    def score(self, x):
        """
        Computes the (difference) score function.
        """
        x = np.atleast_2d(x)
        self.check_valid_input(x)
        n, p = x.shape
        d = int((1 + np.sqrt(1+8*p)) / 2.)  # Number of nodes
        assert 2*p == d*(d-1)

        g_list = [get_graph(d, row) for row in x]

        # Used for retrieving (i, j) indices from x
        row_inds, col_inds = np.triu_indices(d, k=1)

        delta = np.zeros((n, p))
        for s in xrange(n):
            g = g_list[s]  # igraph.Graph
            for k in xrange(p):
                i = row_inds[k]
                j = col_inds[k]
                n_i = get_neighbors(g, i) - {j}
                n_j = get_neighbors(g, j) - {i}

                suff_stats = [1., len(n_i) + len(n_j), len(n_i & n_j)]
                delta[s, k] = np.dot(self.params, suff_stats)

        res = 1. - np.exp(((-1)**x) * delta)  # 0 to 1, 1 to -1

        return res
