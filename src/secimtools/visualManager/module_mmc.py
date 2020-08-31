"""
This is related to the MMC paper.

"""


from functools import partial

import numpy as np
import scipy.linalg
from numpy.testing import assert_equal, assert_allclose


"""
This is a more complicated and specific iterative refinement of a bipartition.

Find a local minimum of a function of two sets.
The domain of the function consists of all possible bipartitions of a set.
Locality is defined with respect to the operation that moves a member
from one set to the other set.

In this module, the cost function is assumed to take a specific form:
the negative of the sum of within-cluster signed edge values.

"""
__all__ = ['faster_refinement']


def _faster_single_refinement(M, alpha, beta):
    """

    Parameters
    ----------
    M : 2d array of edge values
        Larger positive values are evidence of shared cluster membership.
        In particular, the value of the bipartition is the sum of the edge
        values within the first cluster plus the sum of the edge values
        within the second cluster.
        The cost of the bipartition is the negative of its value.
    alpha : set
        First set in the initial bipartition.
        This is a set of indices of M.
    beta : set
        Second set in the initial bipartition.
        This is a set of indices of M.

    Returns
    -------
    initial_cost : float
        Initial cost.
    final_cost : float
        Cost of the maximal bipartition.
    a : set
        First set of the refined bipartition.
    b : set
        Second set of the refined bipartition.

    """
    # Precompute functions of the blocks of M corresponding to the bipartition.
    ia = np.array(list(alpha), dtype=int)
    ib = np.array(list(beta), dtype=int)
    A = M[np.ix_(ia, ia)]
    B = M[np.ix_(ib, ib)]
    AB = M[np.ix_(ia, ib)]
    vaa = A.sum(axis=0)
    vbb = B.sum(axis=0)
    vab = AB.sum(axis=0) + np.diag(B)
    vba = AB.sum(axis=1) + np.diag(A)
    a_move_improvements = vba - vaa
    b_move_improvements = vab - vbb
    initial_cost = -(vaa.sum() + vbb.sum())

    # Check each move from the first set to the second set,
    # and from the second set to the first set.
    best_improvement = 0
    best_pair = alpha, beta
    for i, a in enumerate(ia):
        improvement = a_move_improvements[i]
        if improvement > best_improvement:
            best_improvement = improvement
            best_pair = alpha - {a}, beta | {a}
    for i, b in enumerate(ib):
        improvement = b_move_improvements[i]
        if improvement > best_improvement:
            best_improvement = improvement
            best_pair = alpha | {b}, beta - {b}

    # Return the best improvement and the corresponding bipartition.
    # The 2x factor is related to the fact that the matrix is symmetric.
    best_cost = initial_cost - 2*best_improvement
    return initial_cost, best_cost, best_pair[0], best_pair[1]


def faster_refinement(M, threshold, alpha, beta):
    """
    Repeatedly apply single refinment steps until no improvement is found.

    Parameters
    ----------
    M : 2d array of edge values
        Larger positive values are evidence of shared cluster membership.
    threshold : float
        Cost differences smaller than this value are considered negligible.
    alpha : set
        First set in the initial bipartition.
    beta : set
        Second set in the initial bipartition.

    Returns
    -------
    initial_cost : float
        Initial cost.
    final_cost : float
        Cost of the maximal bipartition.
    a : set
        First set of the maximal bipartition.
    b : set
        Second set of the maximal bipartition.

    """
    initial_cost = None
    cost = None
    a, b = alpha, beta
    while True:
        icost, fcost, na, nb = _faster_single_refinement(M, a, b)
        if initial_cost is None:
            initial_cost = icost
            cost = icost
        if icost - fcost < threshold:
            break
        cost, a, b = fcost, na, nb
    return initial_cost, cost, a, b



def ClusteringError(Exception):
    pass


def max_eigenpair(M):
    # M is a symmetric matrix.
    m = M.shape[0]
    W, V = scipy.linalg.eigh(M, eigvals=(m-1, m-1))
    return W[0], V[:, 0]


def expansion(v):
    # v is an integer vector indicating cluster membership
    v = np.asarray(v, dtype=int)
    k = v.max() + 1
    I = np.eye(k, dtype=int)
    S = np.take(I, v, axis=0)
    return S


def modularity_matrix(A):
    # Also returns the sum of entries of A.
    v = A.sum(axis=1)
    d = v.sum()
    B = A - np.outer(v, v) / d
    return B


def modularity(B, d, v):
    # B is the modularity matrix.
    # d is the sum of entries of the affinity matrix.
    # v is an integer vector indicating cluster membership
    S = expansion(v)
    return np.trace(S.T.dot(B).dot(S)) / d


def modulated_affinity_matrix(R, s):
    # s is a tuning parameter that indirectly controls the number of clusters
    # Note that although the matrix that is returned by this function
    # is hollow with positive off-diagonals, it is not a Euclidean distance
    # matrix (EDM).
    c = 1 / (s*s)
    A = np.exp(c * (np.abs(R) - 1))
    np.fill_diagonal(A, 0)
    return A


def recursive_clustering(B, d, i, clustering, use_refinement):
    # B is the immutable modulated modularity matrix.
    # d is the sum of entries of the affinity matrix.
    # i is an integer indicating the cluster of interest.
    # clustering is the mutable vector of cluster assignments.
    # use_refinement is a flag for iterative refinement after the spectral step.
    p = clustering.shape[0]
    k = clustering.max() + 1
    mask = clustering == i
    selection = np.arange(p, dtype=int)[mask]
    M = B[np.ix_(mask, mask)]
    nosplit_cost = -M.sum()
    M = M - np.diag(M.sum(axis=1))
    w, v = max_eigenpair(M)
    optimality_threshold = 1e-4
    eigen_eps = 1e-8
    if w < eigen_eps:
        return clustering
    m0 = (v < 0)
    m1 = np.logical_not(m0)
    if not np.any(m0) or not np.any(m1):
        return clustering
    sel0 = selection[m0]
    sel1 = selection[m1]
    clustering[sel1] = k

    a, b = set(sel0), set(sel1)
    if use_refinement:
        ci, cf, a, b = faster_refinement(B, optimality_threshold, a, b)
    else:
        B0 = B[np.ix_(sel0, sel0)]
        B1 = B[np.ix_(sel1, sel1)]
        cf = -(B0.sum() + B1.sum())
    new_clustering = clustering.copy()
    new_clustering[list(a)] = i
    new_clustering[list(b)] = k

    if nosplit_cost - cf > optimality_threshold:
        # Use the new clustering provided by the refinement.
        clustering = new_clustering
        ni = (new_clustering == i).size
        nk = (new_clustering == k).size
        if not ni:
            raise ClusteringError
        if not nk:
            raise ClusteringError
        if ni > 1:
            clustering = recursive_clustering(
                    B, d, i, clustering, use_refinement)
        if nk > 1:
            clustering = recursive_clustering(
                    B, d, k, clustering, use_refinement)
    else:
        # Revert the split.
        clustering[sel1] = i
    return clustering


def get_clustering(C, sigmas):
    p = C.shape[0]
    best_clustering = None
    best_sigma = None
    best_m = None
    for i, sigma in enumerate(sigmas):
        A = modulated_affinity_matrix(C, sigma)
        # A is symmetric.
        d = A.sum()
        B = modularity_matrix(A)
        # B is symmetric with row and column sums equal to zero.
        clustering = np.zeros(p, dtype=int)
        # Do not use refinement while search over the values of sigma.
        clustering = recursive_clustering(B, d, 0, clustering, False)
        m = modularity(B, d, clustering)
        if best_m is None or m > best_m:
            best_clustering = clustering
            best_sigma = sigma
            best_m = m
        k = clustering.max() + 1

    # Re-compute the clustering with refinement,
    # using the optimal sigma found without refinement.
    A = modulated_affinity_matrix(C, best_sigma)
    d = A.sum()
    B = modularity_matrix(A)
    clustering = np.zeros(p, dtype=int)
    best_clustering = recursive_clustering(B, d, 0, clustering, True)
    best_m = modularity(B, d, best_clustering)

    # Return the clustering information.
    return best_clustering, best_sigma, best_m


def sample_correlation_matrix(R, n):
    # This is a helper function for reproducing analyses.
    p = R.shape[0]
    X = np.random.multivariate_normal(np.zeros(p), R, size=n)
    # The shape of X is (n, p).
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mu) / std
    C = np.cov(X.T, ddof=0)
    # The diagonal of C should consist of all ones.
    return C


def get_clustering_accuracy(observed_v, desired_v):
    # This is a helper function for reproducing analyses.
    observed_v = np.asarray(observed_v, dtype=int)
    desired_v = np.asarray(desired_v, dtype=int)
    p = desired_v.shape[0]
    S = expansion(desired_v)
    T = expansion(observed_v)
    desired = S.dot(S.T) - np.eye(p)
    desired_compl = (1 - desired) - np.eye(p)
    observed = T.dot(T.T) - np.eye(p)
    observed_compl = (1 - observed) - np.eye(p)
    a = (desired * observed).sum() / desired.sum()
    b = (desired_compl * observed_compl).sum() / desired_compl.sum()
    c = (desired * observed + desired_compl * observed_compl).sum() / (
            (desired + desired_compl).sum())
    return a, b, c


def get_corr_fig2_desired_clustering(expansion_factor=1):
    # This is a helper function for reproducing analyses.
    v = (
            [0] * 2 * expansion_factor +
            [1] * 4 * expansion_factor +
            [2] * 3 * expansion_factor +
            [3] * 3 * expansion_factor)
    return v


def get_corr_fig2(expansion_factor=1):
    # This is a helper function for reproducing analyses.
    v = get_corr_fig2_desired_clustering(expansion_factor)
    S = expansion(v)
    M = np.array([
        [0.9, 0.2,  0.0,  0.0],
        [0.2, 0.7,  0.0,  0.0],
        [0.0, 0.0,  0.6, -0.4],
        [0.0, 0.0, -0.4,  0.8],
        ])
    R = S.dot(M).dot(S.T)
    np.fill_diagonal(R, 1)
    return R
