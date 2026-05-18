"""Utilities shared across NLS solvers."""

import numpy as np


def merge_defaults(opts, defaults):
    """Fill missing keys in *opts* from *defaults* (mutates and returns)."""
    for k, v in defaults.items():
        opts.setdefault(k, v)
    return opts


def gain_ratio(act, pred, F_try):
    """Compute the gain ratio rho = actual_reduction / predicted_reduction."""
    if not np.isfinite(F_try) or not np.isfinite(pred) or pred <= 0:
        return -np.inf
    return act / pred


def diag_jtj(J: np.ndarray) -> np.ndarray:
    """Diagonal of J^T J with safety clipping for non-finite or non-positive entries."""
    d = np.sum(J**2, axis=0)
    d[~np.isfinite(d) | (d <= 0)] = 1.0
    return d


def damping_diag(diag_base: np.ndarray, mode: str) -> np.ndarray:
    """Build damping diagonal from a base diagonal vector.

    Parameters
    ----------
    diag_base : ndarray (n,)
        Base diagonal values (e.g. diag(J^T J) or abs(diag(H))).
    mode : str
        'identity'  -> all ones
        'diagJTJ'   -> use diag_base as-is
        'diagHabs'  -> use diag_base as-is
        'diagJ'     -> (only used by nls_lm_krylov; caller should pre-extract)
        'mix'       -> use diag_base as-is (pre-combined by caller)

    Returns
    -------
    ndarray (n,)  damping diagonal, clipped to >= 1e-15 for non-identity modes.
    """
    if mode == 'identity':
        return np.ones_like(diag_base)
    return np.maximum(diag_base, 1e-15)
