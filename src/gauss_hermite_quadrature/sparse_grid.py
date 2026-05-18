"""Smolyak sparse grid using 1D Gauss-Hermite rules."""

import numpy as np
from math import comb
from itertools import combinations

from .gauss_hermite import _gauss_hermite_raw as _gauss_hermite_1d


def sg_gh_gaussian_sparsegrid(d, lvl):
    """Smolyak sparse grid using 1D Gauss-Hermite rules.

    Approximates ∫ f(x) phi(x) dx ≈ sum_k W(k) * f(X[k]),
    where phi(x) = (2*pi)^(-d/2) * exp(-||x||^2/2).

    Returns nodes X (d x M) and weights W (M,).
    """
    q = lvl + d - 1

    x1d = [None] * q
    w1d = [None] * q
    for lev in range(1, q + 1):
        n = 2**lev - 1
        x, w = _gauss_hermite_1d(n)
        x1d[lev - 1] = x
        w1d[lev - 1] = w

    s_min = max(d, q - d + 1)
    s_max = q

    X_cells = []
    W_cells = []

    for s in range(s_min, s_max + 1):
        c = (-1)**(q - s) * comb(d - 1, q - s)

        for idx in _compositions_pos(s, d):
            idx_arr = [i - 1 for i in idx]
            Xt, Wt = _tensor_rule([x1d[i] for i in idx_arr],
                                  [w1d[i] for i in idx_arr])
            X_cells.append(Xt)
            W_cells.append(c * Wt)

    Xh = np.vstack(X_cells)
    Wh = np.concatenate(W_cells)

    Xh, Wh = _merge_nodes(Xh, Wh)

    X = np.sqrt(2) * Xh
    W = Wh / (np.pi**(d / 2))

    return X.T, W



def _tensor_rule(x_cell, w_cell):
    """Tensor product of per-dimension nodes/weights."""
    d = len(x_cell)
    grids = np.meshgrid(*x_cell, indexing='ij')
    M = grids[0].size
    X = np.zeros((M, d))
    for j in range(d):
        X[:, j] = grids[j].ravel()

    W = w_cell[0].copy()
    for j in range(1, d):
        W = np.kron(w_cell[j], W)
    return X, W


def _compositions_pos(s, d):
    """All positive integer compositions of s into d parts.

    Example: _compositions_pos(4, 2) -> [[1,3], [2,2], [3,1]]

    Parameters
    ----------
    s : int  Sum to partition.
    d : int  Number of parts (each >= 1).

    Returns
    -------
    list of list of int
    """
    if d == 1:
        return [[s]]
    result = []
    for cuts in combinations(range(1, s), d - 1):
        comp = []
        prev = 0
        for c in cuts:
            comp.append(c - prev)
            prev = c
        comp.append(s - prev)
        result.append(comp)
    return result


def _merge_nodes(X, W, tol=1e-12):
    """Merge duplicate nodes in X within tolerance *tol*.

    Weights of merged nodes are summed.

    Parameters
    ----------
    X : ndarray (M, d)  Node locations.
    W : ndarray (M,)    Node weights.
    tol : float         Tolerance for considering nodes duplicate.

    Returns
    -------
    Xu : ndarray (M', d)  Unique nodes.
    Wu : ndarray (M',)    Merged weights.
    """
    Xq = np.round(X / tol) * tol
    _, idx, inv = np.unique(Xq, axis=0, return_index=True, return_inverse=True)
    Wu = np.bincount(inv, weights=W)
    Xu = X[idx]
    return Xu, Wu
