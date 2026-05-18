"""Cholesky-based residual, Jacobian, and exact Hessian computation."""

import numpy as np
from scipy.linalg import cholesky
from scipy.sparse import eye as speye

from .ftrans_chol import _unpack_chol
from .._decomp_utils import triu_col_major


def ftrans_j_chol(objective, x_opt, obs_pts, Ex, Cx, weights_matrix=None):
    """Vectorized residual + Jacobian + exact Hessian for Cholesky parameterization.

    Residual: r_i(x) = f_i * exp(phi_i(x))
    where phi_i = -sum log|U_kk| + 0.5 * ||U * (u - x_i)||^2
    Returns Y, Jacobian, Hessian.
    """
    dimX, N = obs_pts.shape

    W = speye(N) if weights_matrix is None else weights_matrix

    u, U = _unpack_chol(x_opt, dimX)
    u = u.ravel()
    U = np.triu(U)

    du = np.diag(U)
    if np.any(du == 0):
        raise ValueError("U has zero diagonal entry.")

    V = u[:, np.newaxis] - obs_pts

    Ex = Ex.ravel()
    Lc = cholesky(Cx, lower=True)
    X_normalized = Ex[:, np.newaxis] + Lc @ obs_pts
    fval = np.asarray(objective(X_normalized)).ravel()

    A = U @ V
    q = np.sum(A**2, axis=0)
    phi = -np.sum(np.log(np.abs(du))) + 0.5 * q
    phi = np.clip(phi, -700.0, 700.0)
    r = fval * np.exp(phi)
    Y = W @ r

    nUpper = dimX * (dimX + 1) // 2
    nParams = dimX + nUpper

    # Jacobian J0 (N x nParams)
    J0 = np.empty((N, nParams))

    # u block: dr/du = r * (U' * A)^T
    J0[:, :dimX] = (U.T @ A).T * r[:, np.newaxis]

    # U block (column-major triu order)
    col = dimX
    for qcol in range(dimX):
        ncols = qcol + 1
        Ab = A[:qcol + 1, :].T
        G = Ab * V[qcol, :, np.newaxis]
        G[:, qcol] -= 1.0 / U[qcol, qcol]
        J0[:, col:col + ncols] = G * r[:, np.newaxis]
        col += ncols

    Jacobian = W @ J0

    # Exact Hessian of F(x) = 0.5 * ||Y||^2
    Jw = Jacobian
    Hgn = Jw.T @ Jw

    svec = W.T @ Y

    w2 = np.zeros(N)
    ok = (r != 0) & np.isfinite(r) & np.isfinite(svec)
    w2[ok] = svec[ok] / r[ok]

    # J0' @ diag(w2) @ J0 via single einsum call on weighted J0
    sqrt_w2 = np.sqrt(np.maximum(w2, 0.0))
    Jw2 = sqrt_w2[:, np.newaxis] * J0
    Hg = Jw2.T @ Jw2

    beta = svec * r
    beta[~np.isfinite(beta)] = 0
    sumBeta = np.sum(beta)

    Delta = V @ beta
    Gamma = A @ beta
    Mv = (V * beta) @ V.T

    p_idx, q_idx = triu_col_major(dimX)

    Hlog = np.zeros((nParams, nParams))

    # (u, u)
    Hlog[:dimX, :dimX] = sumBeta * (U.T @ U)

    # (u, U)
    Hu1 = np.zeros((dimX, nUpper))
    Hu1[q_idx, np.arange(nUpper)] = Gamma[p_idx]
    Hu2 = U.T[:, p_idx] * Delta[q_idx]
    Hu = Hu1 + Hu2
    Hlog[:dimX, dimX:] = Hu
    Hlog[dimX:, :dimX] = Hu.T

    # (U, U): block-diagonal by row p, vectorized
    idxMap = -np.ones((dimX, dimX), dtype=int)
    idxMap[p_idx, q_idx] = np.arange(nUpper)

    HL = np.zeros((nUpper, nUpper))
    for pRow in range(dimX):
        ids = idxMap[pRow, pRow:]
        ids = ids[ids >= 0]
        k = len(ids)
        if k > 0:
            HL[np.ix_(ids, ids)] += Mv[np.ix_(range(pRow, pRow + k), range(pRow, pRow + k))]
            HL[ids[0], ids[0]] += sumBeta / (U[pRow, pRow]**2)

    Hlog[dimX:, dimX:] = HL

    Hessian = Hgn + Hg + Hlog
    Hessian = 0.5 * (Hessian + Hessian.T)

    return Y, Jacobian, Hessian
