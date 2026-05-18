"""LDL-based residual, Jacobian, and exact Hessian computation."""

import sys
import numpy as np
from scipy.linalg import cholesky
from scipy.sparse import eye as speye

from .ftrans_ldl import _unpack_ldl
from .._decomp_utils import tril_col_major


def ftrans_j_ldl(objective, x_opt, obs_pts, Ex, Cx, weights_matrix=None):
    """Vectorized residual + Jacobian + exact Hessian for LDL parameterization.

    Residual: r_i = f_i * exp(phi_i)
    phi_i = -sum log|d_j| + 0.5 * ||diag(d) * LInv * (u - x_i)||^2
    Returns Y, Jacobian, Hessian.
    """
    dimX, N = obs_pts.shape

    W = speye(N) if weights_matrix is None else weights_matrix

    u, LInv, sqrtDInv = _unpack_ldl(x_opt, dimX)
    u = u.ravel()
    LInv = np.tril(LInv)
    d = np.diag(sqrtDInv)
    if np.any(d == 0):
        raise ValueError("sqrtDInv has zero diagonal entry.")

    Ex = Ex.ravel()
    Lc = cholesky(Cx, lower=True)
    X_normalized = Ex[:, np.newaxis] + Lc @ obs_pts
    fval = np.asarray(objective(X_normalized)).ravel()

    V = u[:, np.newaxis] - obs_pts
    w = LInv @ V
    q = np.sum((d[:, np.newaxis] * w)**2, axis=0)
    phi = -np.sum(np.log(np.abs(d))) + 0.5 * q
    phi = np.clip(phi, -700.0, 700.0)
    r = fval * np.exp(phi)
    Y = W @ r

    nLower = dimX * (dimX - 1) // 2
    nParams = 2 * dimX + nLower

    s = d**2
    T = s[:, np.newaxis] * w

    # Jacobian J0 (N x nParams)
    J0 = np.empty((N, nParams))

    # u block
    J0[:, :dimX] = (LInv.T @ T).T * r[:, np.newaxis]

    # d block
    c = -np.sign(d) / np.maximum(np.abs(d), sys.float_info.min)
    J0[:, dimX:2 * dimX] = ((d[:, np.newaxis] * w**2).T + c) * r[:, np.newaxis]

    # Strict-lower LInv block (column-major ordering)
    col = 2 * dimX
    for qcol in range(dimX - 1):
        ncols = dimX - qcol - 1
        J0[:, col:col + ncols] = (V[qcol, :, np.newaxis] * T[qcol + 1:, :].T) * r[:, np.newaxis]
        col += ncols

    Jacobian = W @ J0

    # Exact Hessian of F(x) = 0.5 * ||Y||^2
    Jw = Jacobian
    Hgn = Jw.T @ Jw

    svec = W.T @ Y

    w2 = np.zeros(N)
    ok = (r != 0) & np.isfinite(r) & np.isfinite(svec)
    w2[ok] = svec[ok] / r[ok]

    sqrt_w2 = np.sqrt(np.maximum(w2, 0.0))
    Jw2 = sqrt_w2[:, np.newaxis] * J0
    Hg = Jw2.T @ Jw2

    beta = svec * r
    beta[~np.isfinite(beta)] = 0
    sumBeta = np.sum(beta)

    gamma = w @ beta
    delta = V @ beta
    eta = (w**2) @ beta
    Cmat = (w * beta) @ V.T

    sqrtBeta = np.sqrt(np.maximum(beta, 0))
    Mv = (V * sqrtBeta) @ (V * sqrtBeta).T

    p_idx, q_idx = tril_col_major(dimX)

    Hlog = np.zeros((nParams, nParams))

    # (u, u)
    A_mat = LInv.T @ (s[:, np.newaxis] * LInv)
    Hlog[:dimX, :dimX] = sumBeta * A_mat

    # (u, d)
    HuD = LInv.T * (2 * d * gamma)
    Hlog[:dimX, dimX:2 * dimX] = HuD
    Hlog[dimX:2 * dimX, :dimX] = HuD.T

    # (d, d)
    Hlog[dimX:2 * dimX, dimX:2 * dimX] = np.diag(sumBeta / d**2 + eta)

    # (u, L)
    HuL1 = np.zeros((dimX, nLower))
    HuL1[q_idx, np.arange(nLower)] = (s * gamma)[p_idx]
    HuL = HuL1 + LInv.T[:, p_idx] * (s[p_idx] * delta[q_idx])
    Hlog[:dimX, 2 * dimX:] = HuL
    Hlog[2 * dimX:, :dimX] = HuL.T

    # (d, L)
    HdL = np.zeros((dimX, nLower))
    HdL[p_idx, np.arange(nLower)] = 2 * d[p_idx] * Cmat[p_idx, q_idx]
    Hlog[dimX:2 * dimX, 2 * dimX:] = HdL
    Hlog[2 * dimX:, dimX:2 * dimX] = HdL.T

    # (L, L): block diagonal by row p, vectorized
    idxMap = -np.ones((dimX, dimX), dtype=int)
    idxMap[p_idx, q_idx] = np.arange(nLower)
    HL = np.zeros((nLower, nLower))
    for pRow in range(1, dimX):
        ids = idxMap[pRow, :pRow]
        ids = ids[ids >= 0]
        k = len(ids)
        if k > 0:
            HL[np.ix_(ids, ids)] += s[pRow] * Mv[np.ix_(range(k), range(k))]

    Hlog[2 * dimX:, 2 * dimX:] = HL

    Hessian = Hgn + Hg + Hlog
    Hessian = 0.5 * (Hessian + Hessian.T)

    return Y, Jacobian, Hessian
