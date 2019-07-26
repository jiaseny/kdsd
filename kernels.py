"""
Collection of kernel functions for discrete data.

Given inputs x and y (n * d arrays), compute the gram matrix K with entries
    K[i, j] = kernel(x[i], y[j])
"""

from __future__ import division
from gk_wl import *  # Graph kernels
from util import *


def hamming_kernel(x, y):
    """
    NOTE: The kernel matrix K is not symmetric, since in general
        K(x[i], y[j]) != K(x[j], y[i])
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    assert x.shape[1] == y.shape[1]  # d

    K = 1. - cdist(x, y, "Hamming")

    assert_shape(K, (x.shape[0], y.shape[0]))

    return K


def exp_hamming_kernel(x, y):
    """
    NOTE: The kernel matrix K is not symmetric, since in general
        K(x[i], y[j]) != K(x[j], y[i])
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    assert x.shape[1] == y.shape[1]  # d

    K = np.exp(-cdist(x, y, "Hamming"))

    assert_shape(K, (x.shape[0], y.shape[0]))

    return K


def wl_kernel_graph(g1_list, g2_list, h=2):
    """
    Computes the Weisfeiler-Lehman graph kernel.

    Args:
        g1_list: list of ig.Graph objects.
        g1_list: list of ig.Graph objects.
        h: int, number of iterations in the W-L algorithm.
    """
    n1 = len(g1_list)
    n2 = len(g2_list)

    g_list = np.concatenate([g1_list, g2_list])
    res = GK_WL(h=h).compare_pairwise(g_list)

    K = res[n1:, :n2]
    assert_shape(K, (n1, n2))

    return K


def wl_kernel(x, y, h=2):
    """
    Computes the Weisfeiler-Lehman graph kernel.

    Args:
        x, y: array((n, p)), n graphs, each row representing the upper-
            triangular entries (excluding diagonal) of the adjacency matrix.
        h: int, number of iterations in the W-L algorithm.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    assert x.shape[1] == y.shape[1]  # d
    p = x.shape[1]  # d*(d-1)/2
    d = int((1 + np.sqrt(1+8*p)) / 2.)  # Number of nodes
    assert 2*p == d*(d-1)

    n1 = x.shape[0]
    n2 = y.shape[0]

    z = np.vstack([x, y])  # (n1 + n2, d)
    g_list = [get_graph(d, row) for row in z]

    res = GK_WL(h=h).compare_pairwise(g_list)

    K = res[n1:, :n2]
    assert_shape(K, (n1, n2))

    return K


def get_graph(d, x):
    """
    Read a graph.

    Args:
        d: int, number of nodes.
        x: array, upper-triangular part of the adjacency matrix.
    """
    assert len(x) == d*(d-1)/2.

    A = np.zeros((d, d))
    A[np.triu_indices(d, k=1)] = x  # Set upper-triangle (excluding diagonal)

    g = ig.Graph.Adjacency(A.tolist(), 'upper')
    assert not g.is_directed()  # Check undirected

    return g
