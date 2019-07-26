from __future__ import division
from util import *


class Ising(object):
    """
    Ising model.
    """
    def __init__(self, d, mu=None, theta=None):
        """
        Args:
            d: int, dimension.
            mu: array(d), node potentials.
            theta: array((d, d)), pair-wise potentials.
        """
        if mu is None:
            mu = np.zeros(d)

        if theta is None:
            theta = np.ones((d, d))

        assert len(mu) == d
        assert theta.shape == (d, d)
        assert is_symmetric(theta)

        self.d = d
        self.mu = deepcopy(mu)
        self.theta = deepcopy(theta)
        self.domain = [-1, 1]

        return

    def __repr__(self):
        return "Ising(d = %d, mu = %s, theta = %s, domain = %s)" % \
            (self.d, self.mu, self.theta, self.domain)

    def __str__(self):
        res = "------------------------------\n"
        res += "Ising(d, mu, theta)\n"
        res += "d = %d\n" % self.d
        res += "mu = %s\n" % np.round(self.mu, 3)
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

        assert x.shape[1] == self.d  # Check dimension
        assert np.all(np.isin(x, self.domain))  # Check values

        return True

    def set_ferromagnet(self, l, temp, periodic=True, anti=False):
        """
        Configure the parameters of the model to an (anti-)ferromagnet:
            mu = 0 for all i
            theta = 1/temp for all i, j (or -1/T for an anti-ferromagnet)

        Args:
            l: int, length of (square-)lattice, d = l^2.
            temp: float, temperature; dist becomes random when temp to infty.
            lattice_periodic: boolean, whether the lattice is periodic.
            anti_ferro: boolean, ferromagnet or anti-ferromagnet.
        """
        if isinstance(l, int):
            dim = [l, l]
        else:
            assert isinstance(l, list)
            dim = l

        d = self.d
        assert np.prod(dim) == d
        # Generate lattice graph
        g = ig.Graph.Lattice(dim=dim, circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()

        mu = np.zeros(d)
        theta = np.ones((d, d)) / temp
        theta = theta * np.triu(A)
        theta = theta + theta.T  # Symmetrize
        assert is_symmetric(theta)

        self.mu = deepcopy(mu)
        self.theta = deepcopy(theta)

        return

    def neg(self, x, i):
        """
        Negate the i-th coordinate of x.
        """
        self.check_valid_input(x)
        x = np.atleast_2d(x)
        assert i < x.shape[1]

        res = deepcopy(x)
        res[:, i] = -res[:, i]  # Flips between +1 and -1

        self.check_valid_input(res)

        return res

    def sample(self, num_iters, num_samples=1, burnin_prop=.5):
        """
        Draw samples from an Ising model via the Metropolis algorithm.

        Returns:
            samples: list of num_samples samples.
            x_hist: all samples after burn-in period.
        """
        # Vectorized implementation: for drawing independent samples only
        d = self.d
        n = num_samples
        num_iters = int(num_iters)
        num_accept = np.zeros(n)

        x = 2 * rand.binomial(n=1, p=.5, size=(n, d)) - 1
        self.check_valid_input(x)

        for t in xrange(num_iters):
            inds = rand.choice(d, size=n)

            b = self.mu[inds] + np.einsum("ij,ij->i", x, self.theta[inds, :])  # (n,)
            probs = np.exp(-2 * x[range(n), inds] * b)
            probs[probs > 1.] = 1.  # Metropolis
            assert_shape(probs, (n,))

            accepted = 1 * (rand.uniform(size=n) < probs)
            signs = 1 - 2 * accepted  # (n,); maps 1 to -1 and 0 to 1
            x[range(n), inds] *= signs  # Flip corresponding bits
            num_accept += accepted  # (n,)

        print "Metropolis acceptance rate: %.4f\n" % \
            (np.mean(num_accept) / num_iters)

        self.check_valid_input(x)

        return x  # (n, d)

    def score(self, x):
        """
        Computes the (difference) score function.
        """
        x = np.atleast_2d(x)
        self.check_valid_input(x)
        n, d = x.shape

        b = np.tile(self.mu, (n, 1)) + x.dot(self.theta)  # (n, d)
        res = 1 - np.exp(-2 * x * b)

        return res

    def plot_grid(self, x, l, figsize=(8, 6)):
        """
        Creates a plot of x on an l-by-l lattice.
        """
        self.check_valid_input(x)
        assert self.d == l**2  # Square grid

        fig = plt.figure(figsize=figsize)
        ax = sns.heatmap(np.reshape(x, (l, l)), square=True,
                         linewidths=1,  # Grid lines
                         xticklabels=False, yticklabels=False,
                         cmap=sns.cubehelix_palette(), cbar=False)

        return fig
