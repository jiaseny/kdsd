"""
Draw samples and apply KDSD and MMD tests to the Bernoulli RBM.
"""
from __future__ import division
from util import *
from kernels import *
from ksd import *
from ksd_utils import *
from mmd import *
from rbm import *


if __name__ == "__main__":
    n, sigma, seed, res_dir = sys.argv[1:]

    n = int(n)  # Sample size
    sigma = float(sigma)  # Linear-spacing
    # sigma = 10.**float(sigma)  # Log-spacing
    seed = int(seed)  # Random seed

    print "n = %d\nsigma=%.1f\nseed=%s\nres_dir=%s\n" % \
        (n, sigma, seed, res_dir)

    rand.seed(seed)

    m = 50  # Number of visible units
    k = 25  # Number of hidden units

    W = rand.normal(size=(m, k)) / m
    bvec = rand.normal(size=(m,))
    cvec = rand.normal(size=(k,))

    # ------------------------- Draw MCMC samples ------------------------- #

    print "Drawing MCMC samples ..."

    model_p = RBM(m, k, W, bvec, cvec)  # Null model
    samples_p, _ = model_p.sample(num_iters=1e5, num_samples=n)

    # Set q to perturbed dist or true p
    true_dist = rand.binomial(n=1, p=.5)  # 0 for p, 1 for q

    # Add Gaussian noise
    W_q = W + rand.normal(loc=0, scale=sigma) if true_dist else W

    model_q = RBM(m, k, W_q, bvec, cvec)
    samples_q, _ = model_q.sample(num_iters=1e5, num_samples=n)

    # ------------------------- Perform KDSD test ------------------------- #

    print "Performing KDSD test ..."

    model = RBM(m, k, W, bvec, cvec)  # Null model
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

    res = {'n': n, 'sigma': sigma,
           'true_dist': true_dist,
           'ksd_stat': ksd_stat, 'ksd_thres': ksd_thres,
           'ksd_pval': ksd_pval, 'ksd_pred': ksd_pred,
           'mmd_stat': mmd_stat, 'mmd_thres': mmd_thres,
           'mmd_pval': mmd_pval, 'mmd_pred': mmd_pred}

    pckl_write(res, res_dir + "pow-rbm%.3f-n%d-m%d-k%d-seed%d.res" %
               (sigma, n, m, k, seed))

    print 'Finished!'
