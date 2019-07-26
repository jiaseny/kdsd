"""
Apply KDSD and MMD tests to the ERGM.
Samples are drawn using ergm-sample.R and are saved as text files.
"""
from __future__ import division
from util import *
from kernels import *
from ksd import KSD
from ksd_utils import *
from mmd import MMD
from ergm import ERGM


if __name__ == "__main__":
    n, d, rho0, theta0, tau0, rho, theta, tau, seed, res_dir = sys.argv[1:]

    n = int(n)  # Sample size
    d = int(d)  # Number of nodes
    rho0 = float(rho0)
    theta0 = float(theta0)
    tau0 = float(tau0)
    rho = float(rho)
    theta = float(theta)
    tau = float(tau)
    seed = int(seed)  # Random seed

    print "n = %d\nd = %d\nrho = %.3f\ntheta = %.3f\ntau = %.3f\nres_dir=%s\n" % \
        (n, d, rho, theta, tau, seed res_dir)

    # ------------------------ Set null model params ------------------------ #

    rand.seed(seed)

    # Set q to perturbed dist or true p
    true_dist = rand.binomial(n=1, p=.5)  # 0 for p, 1 for q

    print "true_dist = %d" % true_dist

    print "Null model params:"
    print "rho = %.3f\ntheta = %.3f\ntau = %.3f\n" % (rho0, theta0, tau0)

    print "Model q params:"
    print "rho = %.3f\ntheta = %.3f\ntau = %.3f\n" % (rho, theta, tau)

    # ------------------------- Load samples ------------------------- #

    print "Loading samples ..."

    samples0 = np.loadtxt(res_dir + "ergm-samples-n%d-d%d-rho%.3f-theta%.3f-tau%.3f-seed%d.txt" %
                          (n, d, rho0, theta0, tau0, seed))
    assert samples0.shape[0] == 2 * n, samples0.shape

    # Get samples for q
    if true_dist:  # q dist
        samples1 = np.loadtxt(res_dir + "ergm-samples-n%d-d%d-rho%.3f-theta%.3f-tau%.3f-seed%d.txt" %
                              (n, d, rho, theta, tau, seed))
        assert samples1.shape[0] == 2 * n, samples1.shape

        samples_q = samples1[:n, :]
    else:  # p dist
        samples_q = samples0[n:, :]

    # Get samples for p
    samples_p = samples0[:n, :]

    assert samples_p.shape == samples_q.shape
    assert len(samples_p) == n

    # ------------------------- Perform KDSD test ------------------------- #

    print "Performing KDSD test ..."

    model = ERGM(d, rho=rho0, theta=theta0, tau=tau0)  # Null model
    ksd = KSD(neg_fun=model.neg, score_fun=model.score,
              kernel_fun=wl_kernel)  # Use null model
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

    mmd = MMD(kernel_fun=wl_kernel)
    mmd_stat, mmd_thres, mmd_pval, _ = mmd.perform_test(samples_p, samples_q)
    mmd_pred = 1 * (mmd_stat > mmd_thres)  # 0 for p, 1 for q

    res = {'n': n, 'param0': [rho0, theta0, tau0], 'param': [rho, theta, tau],
           'true_dist': true_dist,
           'ksd_stat': ksd_stat, 'ksd_thres': ksd_thres,
           'ksd_pval': ksd_pval, 'ksd_pred': ksd_pred,
           'mmd_stat': mmd_stat, 'mmd_thres': mmd_thres,
           'mmd_pval': mmd_pval, 'mmd_pred': mmd_pred}

    pckl_write(res, res_dir + "pow-ergm-n%d-d%d-rho%.3f-theta%.3f-tau%.3f-seed%d.res" %
               (n, d, rho, theta, tau, seed))

    print 'Finished!'
