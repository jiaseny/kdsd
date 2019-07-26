from __future__ import division

execfile('util.py')


class MMD(object):
    """
    Quadratic-time maximum mean discrepancy (MMD) test.

    Use the unbiased U-statistic.
    """
    def __init__(self, kernel_fun, use_ustat=True, quantile=.95, n_boot=5000):
        """
        Args:
            kernel: function, kernel function.
            use_ustat: boolean, whether to compute U-statistic or V-statistic.
            quantile: float, 1. - significance level
            n_boot: int, number of bootstraps for computing threshold
        """
        assert callable(kernel_fun)

        self.kernel = kernel_fun
        self.quantile = quantile  # 1 - alpha
        self.n_boot = n_boot  # Number of bootstraps for computing threshold
        self.use_ustat = use_ustat

        return

    def compute_gram(self, x, y):
        """
        Compute Gram matrices:
            K: array((m+n, m+n))
            kxx: array((m, m))
            kyy: array((n, n))
            kxy: array((m, n))
        """
        (m, d1) = x.shape
        (n, d2) = y.shape
        assert d1 == d2

        xy = np.vstack([x, y])
        K = self.kernel(xy, xy)  # kxyxy
        assert_shape(K, (m+n, m+n))
        assert is_psd(K)  # TODO: Remove check

        kxx = K[:m, :m]
        assert_shape(kxx, (m, m))
        # assert is_psd(kxx)
        assert is_symmetric(kxx)

        kyy = K[m:, m:]
        assert_shape(kyy, (n, n))
        # assert is_psd(kyy)
        assert is_symmetric(kyy)

        kxy = K[:m, m:]
        assert_shape(kxy, (m, n))

        return K, kxx, kyy, kxy

    def compute_statistic(self, kxx, kyy, kxy):
        """
        Compute MMD test statistic.
        """
        m = kxx.shape[0]
        n = kyy.shape[0]
        assert_shape(kxx, (m, m))
        assert_shape(kyy, (n, n))
        assert_shape(kxy, (m, n))

        if self.use_ustat:  # Compute U-statistics estimate
            term_xx = (kxx.sum() - np.diag(kxx).sum()) / (m*(m-1))
            term_yy = (kyy.sum() - np.diag(kyy).sum()) / (n*(n-1))
            term_xy = kxy.sum() / (m*n)

        else:  # Compute V-statistics estimate
            term_xx = kxx.sum() / (m**2)
            term_yy = kyy.sum() / (n**2)
            term_xy = kxy.sum() / (m*n)

        res = term_xx + term_yy - 2*term_xy

        return res

    def compute_threshold(self, m, n, K):
        """
        Compute test threshold via bootstrapping.
        """
        assert_shape(K, (m+n, m+n))  # Full kernel matrix

        boot_stats = np.zeros(self.n_boot)

        for t in xrange(self.n_boot):
            inds = rand.choice(m+n, m+n, replace=False)

            # Split into new data indices
            inds_x = inds[:m]
            inds_y = inds[m:]

            # Index via cross-product
            kxx = K[np.ix_(inds_x, inds_x)]
            kyy = K[np.ix_(inds_y, inds_y)]
            kxy = K[np.ix_(inds_x, inds_y)]

            assert_shape(kxx, (m, m))
            assert_shape(kxy, (m, n))
            assert_shape(kyy, (n, n))

            boot_stats[t] = self.compute_statistic(kxx, kyy, kxy)

        thres = np.percentile(boot_stats, 100.*self.quantile)

        return thres, boot_stats

    def compute_pval(self, stat, boot_stats):
        """
        Computes the p-value of the test.
        """

        return np.mean(boot_stats >= stat)

    def perform_test(self, x, y):
        """
        Overall function to perform the test.
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        (m, d1) = x.shape
        (n, d2) = y.shape
        assert d1 == d2

        print "\nComputing Gram matrices ..."
        K, kxx, kyy, kxy = self.compute_gram(x, y)

        print "\nComputing MMD test statistic ..."
        stat = self.compute_statistic(kxx, kyy, kxy)

        print "\nComputing MMD test threshold via bootstrapping ..."
        thres, boot_stats = self.compute_threshold(m, n, K)

        pval = self.compute_pval(stat, boot_stats)

        return stat, thres, pval, boot_stats
