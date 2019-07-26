"""
Draw samples and apply KDSD and MMD tests to the Ising model.
"""
from __future__ import division
from util import *
from kernels import *
from ksd import KSD
from ksd_utils import *
from mmd import MMD
from ising import Ising


if __name__ == "__main__":
    n, temp0, temp, seed, res_dir = sys.argv[1:]

    n = int(n)  # Sample size
    temp0 = float(temp0)  # Temperature under p (H0)
    temp = float(temp)  # Temperature under q
    seed = int(seed)  # Random seed

    print "n = %d\ntemp0 = %s\ntemp = %d\nseed = %s\nres_dir=%s\n" % \
        (n, temp0, temp, seed, res_dir)

    rand.seed(seed)

    l = 10  # Dimension of 2D-lattice
    d = l ** 2  # Dimension of random vector

    # ------------------------- Draw MCMC samples ------------------------- #

    print "Drawing MCMC samples ..."

    model_p = Ising(d)
    model_p.set_ferromagnet(l, temp0)
    samples_p = model_p.sample(num_iters=1E5, num_samples=n)

    # Set q to perturbed dist or true p
    true_dist = rand.binomial(n=1, p=.5)  # 0 for p, 1 for q

    model_q = Ising(d)
    temp_q = temp if true_dist else temp0
    model_q.set_ferromagnet(l, temp_q)
    samples_q = model_q.sample(num_iters=1E5, num_samples=n)

    # ------------------------- Perform KDSD test ------------------------- #

    print "Performing KDSD test ..."

    model = Ising(d)
    model.set_ferromagnet(l, temp0)
    ksd = KSD(neg_fun=model.neg, score_fun=model.score,
              kernel_fun=exp_hamming_kernel)  # Use null model
    kappa_vals = ksd.compute_kappa(samples=samples_q)

    # Compute U-statistics and bootstrap intervals
    ksd_stats, _ = ksd_est([kappa_vals])
    ksd_boot_list, ksd_thres_list = ksd_boot([kappa_vals])
    ksd_pvals = ksd_pvalue(ksd_boot_list, ksd_stats)

    ksd_stat = ksd_stats[0]
    ksd_thres = ksd_thres_list[0]
    ksd_pval = ksd_pvals[0]
    ksd_pred = 1 * (ksd_stat > ksd_thres)  # 0 for p, 1 for q

    # ------------------------- Perform MMD test ------------------------- #

    print "Performing MMD test ..."

    mmd = MMD(kernel_fun=exp_hamming_kernel)
    mmd_stat, mmd_thres, mmd_pval, _ = mmd.perform_test(samples_p, samples_q)
    mmd_pred = 1 * (mmd_stat > mmd_thres)  # 0 for p, 1 for q

    res = {'n': n, 'temp0': temp0, 'temp': temp,
           'true_dist': true_dist,
           'ksd_stat': ksd_stat, 'ksd_thres': ksd_thres,
           'ksd_pval': ksd_pval, 'ksd_pred': ksd_pred,
           'mmd_stat': mmd_stat, 'mmd_thres': mmd_thres,
           'mmd_pval': mmd_pval, 'mmd_pred': mmd_pred}

    pckl_write(res, res_dir + "pow-ising-l%d-T0%s-T%s-n%d-seed%d.res" %
               (l, temp0, temp, n, seed))

    print 'Finished!'
