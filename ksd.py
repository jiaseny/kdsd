from __future__ import division
from util import *

# n: number of samples
# d: dim of features
# x: array(n, d)


class KSD(object):
    """
    Kernelized discrete Stein discrepancy.
    """
    def __init__(self, neg_fun, score_fun, kernel_fun, neg_inv_fun=None):
        """
        Args:
            d: int, input dimension.
            neg_fun: function, cyclic permutation.
            score_fun: function, score function for a model.
            kernel: function, kernel function.
            neg_fun_inv: function, inverse cyclic permutation.
        """
        if neg_inv_fun is None:  # For binary distributions
            neg_inv_fun = neg_fun

        assert callable(neg_fun)
        assert callable(neg_inv_fun)
        assert callable(score_fun)
        assert callable(kernel_fun)

        self.neg = neg_fun
        self.neg_inv = neg_inv_fun
        self.score = score_fun
        self.kernel = kernel_fun

        return

    def diff(self, f, x, inv=False):
        """
        Computes the finite-difference of a function at x:
            diff f(x) = ( f(x) - f(neg_i x) )
            where neg is replaced by neg_inv if inv is True.

        Args:
            f: function (possibly vector-valued).
            x: array of length d.
            inv: boolean, whether to compute diff w.r.t. neg or neg_inv.

        Returns:
            diff f(x): array of shape (d,).
        """
        assert callable(f)
        neg = self.neg if not inv else self.neg_inv  # Cyclic permutation

        x = np.atleast_2d(x)
        n, d = x.shape
        val = f(x)

        res = np.zeros((n, d))
        for i in xrange(d):
            res[:, i] = val - f(neg(x, i))

        return res

    def kernel_temp(self, x):
        """
        Compute intermediate kernel results.

        Returns:
            kxx: array((n, n)), kernel matrix.
            k_x: array((n, n, d)), self.kernel(self.neg(x, l), x)
            k_x_x: array((n, n, d)), self.kernel(x, self.neg(x, l))
        """
        x = np.atleast_2d(x)
        n, d = x.shape

        # Vectorized implementation
        kxx = self.kernel(x, x)  # (n, n)
        assert_shape(kxx, (n, n))

        k_xx = np.zeros((n, n, d))
        k_x_x = np.zeros((n, n, d))

        for l in xrange(d):
            if l % 100 == 0:
                print "\tkxx, k_xx, k_x_x: l = %d ..." % l

            neg_l_x = self.neg_inv(x, l)
            k_xx[:, :, l] = self.kernel(neg_l_x, x)
            k_x_x[:, :, l] = self.kernel(neg_l_x, neg_l_x)

        return [kxx, k_xx, k_x_x]

    def kernel_diff(self, x, kernel_res, arg):
        """
        Computes diff_x k(x, y) if arg == 0, and diff_y k(x, y) if arg == 1.

        Args:
            kernel: kernel function.
            x, y: arrays of length d.

        Returns:
            array of length d.
        """
        x = np.atleast_2d(x)
        n, d = x.shape

        kxx, k_xx, k_x_x = kernel_res

        assert_shape(kxx, (n, n))
        assert_shape(k_xx, (n, n, d))
        assert_shape(k_x_x, (n, n, d))

        if arg == 0:
            res = kxx[:, :, np.newaxis] - k_xx

        elif arg == 1:
            res = kxx[:, :, np.newaxis] - k_xx.swapaxes(0, 1)

        else:
            raise ValueError("arg = %d not recognized!" % arg)

        return res

    def kernel_diff2_tr(self, x, kernel_res):
        """
        Computes trace( diff_x diff_y k(x, y) ).

        Args:
            kernel: kernel function.
            kernel_res: tuple of arrays, see kernel_temp() output.

        Returns:
            array((n, n)), trace value for each x[i] and y[j].
        """
        x = np.atleast_2d(x)

        n = x.shape[0]
        d = x.shape[1]

        kxx, k_xx, k_x_x = kernel_res

        assert_shape(kxx, (n, n))
        assert_shape(k_xx, (n, n, d))
        assert_shape(k_x_x, (n, n, d))

        k_xx_tr = np.sum(k_xx, axis=-1)
        k_x_x_tr = np.sum(k_x_x, axis=-1)

        res = kxx*d - k_xx_tr - k_xx_tr.T + k_x_x_tr  # (n, n)

        return res

    def kappa(self, x):
        """
        Computes the KSD kappa matrix.
        """
        kernel_mat = self.kernel(x, x)  # (n, n)
        assert is_symmetric(kernel_mat)
        score_mat = self.score(x)  # (n, d)

        print "\nComputing kxx, k_xx, k_x_x ..."  # Heavy
        kernel_res = self.kernel_temp(x)

        print "\nComputing kernel_diff ..."

        kdiff_mat = self.kernel_diff(x, kernel_res, arg=1)  # (n, n, d)

        term1 = score_mat.dot(score_mat.T) * kernel_mat
        assert is_symmetric(term1)

        term2 = np.einsum("ik,ijk->ij", score_mat, kdiff_mat)  # (n, n)

        term3 = term2.T

        print "\nComputing kernel_diff2_tr ..."

        term4 = self.kernel_diff2_tr(x, kernel_res)  # (n, n)
        assert is_symmetric(term4)

        res = term1 - term2 - term3 + term4

        return res

    def compute_kappa(self, samples):
        """
        Compute the KSD kernel matrix kappa_p.

        Args:
            samples: array((n, d)).

        Returns:
            kappa_vals: array((n, n)), computed KSD kernel matrix.
        """
        assert isinstance(samples, np.ndarray)
        assert len(samples.shape) == 2

        kappa_vals = self.kappa(samples)

        return kappa_vals
