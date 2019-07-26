from __future__ import division
from bisect import bisect, bisect_left
import cPickle
import igraph as ig
import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import ceil, floor
import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import sys
import warnings
from collections import Counter
from copy import deepcopy
from itertools import chain, izip
from matplotlib.colors import LogNorm
from numpy import random as rand
from scipy.linalg import eigvals, norm, expm, svd, pinv
from operator import itemgetter
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags, identity, \
    issparse
from scipy.spatial.distance import cdist
from scipy.stats import chisquare, gaussian_kde, probplot
from scipy.stats import gamma as gamma_dist
from scipy.special import logsumexp
from scipy.optimize import minimize, minimize_scalar, check_grad
from sklearn.metrics import roc_curve, auc

_EPS = 1e-6
_INF = 1e6

sns.set_style('whitegrid')


# --------------------------------------------------------------------------- #
# Simple utility functions
# --------------------------------------------------------------------------- #

def assert_eq(a, b, message=""):
    """Check if a and b are equal."""
    assert a == b, "Error: %s != %s ! %s" % (a, b, message)
    return


def assert_le(a, b, message=""):
    """Check if a and b are equal."""
    assert a <= b, "Error: %s > %s ! %s" % (a, b, message)
    return


def assert_ge(a, b, message=""):
    """Check if a and b are equal."""
    assert a >= b, "Error: %s < %s ! %s" % (a, b, message)
    return


def assert_len(l, length, message=""):
    """Check list/array l is of shape shape."""
    assert_eq(len(l), length)
    return


def assert_shape(A, shape, message=""):
    """Check array A is of shape shape."""
    assert_eq(A.shape, shape)
    return


def check_prob_vector(p):
    """
    Check if a vector is a probability vector.

    Args:
        p, array/list.
    """
    assert np.all(p >= 0), p
    assert np.isclose(np.sum(p), 1), p

    return True


def del_inds(l, inds):
    """
    Delete elements from l indexed by a list of arbitrary indices.
    Operates in place by modifying original list l.

    Args:
        l, original list.
        inds, list of indices to be deleted.
    """
    assert isinstance(l, list) and isinstance(inds, list)

    for idx in sorted(inds, reverse=True):
        del l[idx]

    return


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Check approximate equality for floating-point numbers."""
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def is_unique(l):
    """Check if all the elements in list l are unique."""
    assert isinstance(l, list), "Type %s is not list!" % type(l)
    return len(l) == len(set(l))


def is_sorted(l):
    """Check if all the elements in list l are sorted in ascending order."""
    return all(l[i] <= l[i+1] for i in xrange(len(l)-1))


def is_symmetric(A):
    """Check if A is a symmetric matrix."""
    assert isinstance(A, np.ndarray)
    return norm(A - A.T) < _EPS


def rand_choice(d, size=None):
    """Convenience wrapper for rand.choice to draw from a probability dist."""
    assert isinstance(d, Counter) or isinstance(d, dict)
    assert is_close(sum(d.values()), 1.)  # Properly normalized
    return rand.choice(a=d.keys(), p=d.values(), size=size)


def flatten(l, r=1):
    """Flatten a nested list/tuple r times."""
    return l if r == 0 else flatten([e for s in l for e in s], r-1)


def binarize(x):
    assert isinstance(x, np.ndarray)
    return 1 * (x > 0)


def deduplicate(seq):
    """Remove duplicates from list/array while preserving order."""
    seen = set()
    seen_add = seen.add  # For efficiency due to dynamic typing
    return [e for e in seq if not (e in seen or seen_add(e))]


def arg_max(l):
    """
    Probabilistic version of np.argmax() in which the arg-max is drawn
        uniformly at random from the set of maximum elements.
    """
    assert isinstance(l, list) or isinstance(l, np.ndarray)
    return rand.choice(np.flatnonzero(l == np.max(l)))


def most_common(cnt):
    """
    Probabilistic version of Counter.most_common() in which the most-common
    element is drawn uniformly at random from the set of most-common elements.
    """
    assert isinstance(cnt, Counter) or isinstance(cnt, dict)

    max_count = max(cnt.values())
    most_common_elements = [i for i in cnt.keys() if cnt[i] == max_count]
    return rand.choice(most_common_elements)


def nnz(A):
    """
    Returns the number of non-zero elements in A.
    """
    assert isinstance(A, np).ndarray

    return np.sum(A != 0)


def density(A):
    """
    Computes the density of A.
    """
    assert isinstance(A, np).ndarray

    return nnz(A) / np.prod(A.shape)


def normalize(counts, alpha=0.):
    """
    Normalize counts to produce a valid probability distribution.

    Args:
        counts: A Counter/dict/np.ndarray/list storing un-normalized counts.
        alpha: Smoothing parameter (alpha = 0: no smoothing;
            alpha = 1: Laplace smoothing).

    Returns:
        A Counter/np.array of normalized probabilites.
    """
    if isinstance(counts, Counter):
        # Returns the normalized counter without modifying the original one
        temp = sum(counts.values()) + alpha * len(counts.keys())
        dist = Counter({key: (counts[key]+alpha) / temp
                        for key in counts.keys()})
        return dist

    elif isinstance(counts, dict):
        # Returns the normalized dict without modifying the original one
        temp = sum(counts.values()) + alpha * len(counts.keys())
        dist = {key: (counts[key]+alpha) / temp for key in counts.keys()}

        return dist

    elif isinstance(counts, np.ndarray):
        temp = sum(counts) + alpha * len(counts)
        dist = (counts+alpha) / temp
        check_prob_vector(dist)

        return dist

    elif isinstance(counts, list):
        return normalize(np.array(counts))

    else:
        raise ValueError("Input type %s not understood!" % type(counts))

    return


def is_psd(A):
    """
    Check if A is p.s.d. by computing the minimum eigenvalue of the np.array A.
    """
    if not is_symmetric(A):
        warnings.warn("Matrix not symmetric!")

    min_eigval = np.min(np.linalg.eigvals(A))
    if min_eigval < -_EPS:
        warnings.warn("Minimum eigenvalue is %s < 0!" % min_eigval)
        return False

    # chol = np.linalg.cholesky(A)  # Symmetric matrix has Cholesky iff. psd

    return True


def normalize_rows(A):
    """
    Normalize the rows of array A such that each row sums to 1.
    """
    assert isinstance(A, np.ndarray)
    assert_len(A.shape, 2)  # Check 2-D array

    row_sums = A.sum(axis=1)
    res = A / row_sums[:, np.newaxis]

    check_row_stochastic(res)

    return res


def normalize_cols(A):
    """
    Normalize the rows of array A such that each row sums to 1.
    """
    assert isinstance(A, np.ndarray)
    assert_len(A.shape, 2)  # Check 2-D array

    row_sums = A.sum(axis=0)
    res = A / row_sums[np.newaxis, :]

    check_row_stochastic(res)

    return res


def accuracy(pred_labels, true_labels):
    """
    Computes binary classification accuracy.

    Args:
    """
    assert len(pred_labels) == len(true_labels)
    num = len(pred_labels)
    num_correct = sum([pred_labels[i] == true_labels[i] for i in xrange(num)])

    acc = num_correct / float(num)

    return acc


def sigmoid(z):
    """Logistic sigmoid function."""
    return 1. / (1 + np.exp(-z))


def rand_bern(p=.5, size=1):
    """Generate Bernoulli random numbers."""
    return rand.binomial(n=1, p=p, size=size)


def rand_multinomial(p):
    """Draw a multinomial r.v. with probability vector p."""
    check_prob_vector(p)

    return rand.choice(a=len(p), p=p)


def rand_multinomial_log(log_p):
    """Draw a multinomial r.v. with log-probability vector log_p."""
    p = np.exp(log_p - logsumexp(log_p))

    return rand_multinomial(p)


def get_sparse_adjacency(graph, weight_attr=None):
    """
    Returns the sparse adjacency matrix for igraph graph.
    """
    assert isinstance(graph, ig.Graph) or isinstance(graph, DataGraph)

    edges = graph.get_edgelist()

    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]

    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)

    return csr_matrix((weights, zip(*edges)))


def get_neighbors(g, i):
    """
    Returns the indices of the neighbors of nodes i in igraph g.
    """
    assert isinstance(g, ig.Graph)
    assert isinstance(i, int)

    return set(v.index for v in g.vs[i].neighbors())


def get_triu_index(n, i, j, k=1):
    assert j >= i + k
    ind = j - i - k

    if i > 0:
        ind += (n-k) * i - i*(i-1)//2

    assert ind >= 0

    return ind


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def pckl_write(data, filename):
    with open(filename, 'w') as f:
        cPickle.dump(data, f)

    return


def pckl_read(filename):
    with open(filename, 'r') as f:
        data = cPickle.load(f)

    return data
