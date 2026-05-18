"""Shared utilities for Cholesky and LDL decomposition parameterizations."""

import numpy as np


def triu_col_major(n):
    """Return (p_idx, q_idx) for upper triangle in column-major (MATLAB) order."""
    p_idx, q_idx = np.triu_indices(n)
    order = np.lexsort((p_idx, q_idx))
    return p_idx[order], q_idx[order]


def tril_col_major(n):
    """Return (p_idx, q_idx) for strict lower triangle in column-major order."""
    p_idx, q_idx = np.tril_indices(n, -1)
    order = np.lexsort((p_idx, q_idx))
    return p_idx[order], q_idx[order]
