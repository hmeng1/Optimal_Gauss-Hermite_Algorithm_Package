"""Cholesky-based forward transform and parameter packing/unpacking."""

import numpy as np
from scipy.linalg import cholesky

from .._decomp_utils import triu_col_major


def ftrans_chol(objective, x_opt, obs_pts, Ex, Cx):
    """Compute the unweighted transform value for Cholesky parameterization.

    x_opt = [u; vecu(U)] where U is upper-triangular column-major.
    Model: Sigma^{-1} = U' * U.
    Returns y (N,).
    """
    dimX = obs_pts.shape[0]

    u, U = _unpack_chol(x_opt, dimX)
    u = u.ravel()
    U = np.triu(U)

    dU = np.diag(U)
    if np.any(dU == 0):
        raise ValueError("U has zero diagonal entry.")

    # Z = U^{-1} * obs_pts (upper-triangular back-substitution)
    Z = _upper_solve(U, obs_pts)
    X = u[:, np.newaxis] + Z

    Ex = Ex.ravel()
    Lc = cholesky(Cx, lower=True)
    X_normalized = Ex[:, np.newaxis] + Lc @ X

    fval = objective(X_normalized)
    fval = np.asarray(fval).ravel()

    B = X - u[:, np.newaxis]
    UB = U @ B
    qDen = np.sum(UB**2, axis=0)

    qNum = np.sum(X**2, axis=0)

    logDetTerm = np.sum(np.log(np.abs(dU)))
    logRatio = np.clip(-logDetTerm - 0.5 * (qNum - qDen), -700.0, 700.0)
    y = fval * np.exp(logRatio)
    return y


def _unpack_chol(x, n):
    """Unpack x = [u; vecu(U)] into u (n,) and U (n x n upper-triangular).

    vecu(U) stacks nonzero entries of U column-by-column (MATLAB column-major triu order).
    """
    x = np.asarray(x).ravel()
    nUpper = n * (n + 1) // 2
    nExpected = n + nUpper
    if len(x) != nExpected:
        raise ValueError(f"Expected length {nExpected} for dimX={n}, got {len(x)}.")

    u = x[:n]
    uvec = x[n:]

    U = np.zeros((n, n))
    # Column-major triu indices
    p_idx, q_idx = triu_col_major(n)
    U[p_idx, q_idx] = uvec
    return u, U


def _pack_chol(u, C):
    """Pack parameters into x = [u; vecu(U)] where U = chol(C) upper-triangular."""
    u = np.asarray(u).ravel()
    n = len(u)

    U = np.linalg.cholesky(C).T  # scipy returns lower, transpose for upper
    U = np.triu(U)

    p_idx, q_idx = triu_col_major(n)
    uvec = U[p_idx, q_idx]

    return np.concatenate([u, uvec])


def _upper_solve(U, B):
    """Solve U @ X = B for upper-triangular U via back-substitution.

    Parameters
    ----------
    U : ndarray (n, n)
        Upper-triangular matrix (only upper triangle used).
    B : ndarray (n, N)
        Right-hand sides.

    Returns
    -------
    ndarray (n, N)
        Solution X such that U @ X = B.
    """
    U = np.triu(U)
    X = np.array(B, dtype=float)
    n = U.shape[0]
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            X[i, :] = X[i, :] - U[i, i + 1:] @ X[i + 1:, :]
        X[i, :] = X[i, :] / U[i, i]
    return X
