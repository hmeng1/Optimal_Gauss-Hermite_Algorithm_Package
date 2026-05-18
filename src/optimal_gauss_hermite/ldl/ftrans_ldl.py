"""LDL-based forward transform and parameter packing/unpacking."""

import numpy as np
from scipy.linalg import cholesky

from .._decomp_utils import tril_col_major


def ftrans_ldl(objective, x_opt, obs_pts, Ex, Cx):
    """Compute the unweighted transform value for LDL parameterization.

    x_opt = [u; diag(sqrtDInv); vech_strict(LInv)].
    Model: uses LInv (unit lower-tri) and sqrtDInv (diagonal).
    Returns y (N,).
    """
    dimX = obs_pts.shape[0]

    u, LInv, sqrtDInv = _unpack_ldl(x_opt, dimX)
    u = u.ravel()
    LInv = np.tril(LInv)
    d = np.diag(sqrtDInv)

    # Structured solve: X = u + (sqrtDInv * LInv) \ obs_pts
    rhs = obs_pts / d[:, np.newaxis]
    Z = _unit_lower_solve(LInv, rhs)
    X = u[:, np.newaxis] + Z

    Ex = Ex.ravel()
    Lc = cholesky(Cx, lower=True)
    X_normalized = Ex[:, np.newaxis] + Lc @ X

    fval = objective(X_normalized)
    fval = np.asarray(fval).ravel()

    B = X - u[:, np.newaxis]
    UL = LInv @ B
    Zden = d[:, np.newaxis] * UL
    qDen = np.sum(Zden**2, axis=0)

    qNum = np.sum(X**2, axis=0)

    logDetTerm = np.sum(np.log(np.abs(d)))
    logRatio = np.clip(-logDetTerm - 0.5 * (qNum - qDen), -700.0, 700.0)
    y = fval * np.exp(logRatio)
    return y


def _unpack_ldl(x, dimX):
    """Unpack x = [u; diag(D); strictly-lower(L)]."""
    x = np.asarray(x).ravel()
    nLower = dimX * (dimX - 1) // 2
    nExpected = 2 * dimX + nLower
    if len(x) != nExpected:
        raise ValueError(f"Expected length {nExpected} for dimX={dimX}, got {len(x)}.")

    u = x[0:dimX]
    d = x[dimX:2 * dimX]
    D = np.diag(d)

    L = np.eye(dimX)
    p_idx, q_idx = tril_col_major(dimX)
    L[p_idx, q_idx] = x[2 * dimX:2 * dimX + nLower]

    return u, L, D


def _pack_ldl(u, C):
    """Pack parameters into x = [u; sqrtD; vecl_strict(L)]."""
    u = np.asarray(u).ravel()
    n = len(u)

    from scipy.linalg import ldl as scipy_ldl
    LU, D, _ = scipy_ldl(C)
    L = np.tril(LU)
    d = np.diag(D)

    if np.any(d <= 0):
        raise ValueError("Diagonal of D has nonpositive entries.")

    sqrtD = np.sqrt(d)

    p_idx, q_idx = tril_col_major(n)
    lvec = L[p_idx, q_idx]

    return np.concatenate([u, sqrtD, lvec])


def _unit_lower_solve(L, B):
    """Solve L @ X = B for unit-lower-triangular L via forward substitution.

    L must have ones on its diagonal (only strict lower triangle is used).

    Parameters
    ----------
    L : ndarray (n, n)
        Unit lower-triangular matrix.
    B : ndarray (n, N)
        Right-hand sides.

    Returns
    -------
    ndarray (n, N)
        Solution X such that L @ X = B.
    """
    L = np.tril(L)
    X = np.array(B, dtype=float)
    n = L.shape[0]
    for i in range(n):
        if i > 0:
            X[i, :] = X[i, :] - L[i, :i] @ X[:i, :]
    return X
