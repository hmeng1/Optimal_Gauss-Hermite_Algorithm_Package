"""1D Gauss-Hermite quadrature nodes and weights via the Golub-Welsch algorithm."""

import numpy as np


def _gauss_hermite_raw(n):
    """Raw Gauss-Hermite nodes x and weights w for weight exp(-x^2).

    Returns nodes of the Jacobi matrix (no sqrt(2) scaling) and
    unnormalized weights summing to sqrt(pi).
    """
    if n == 1:
        return np.array([0.0]), np.array([np.sqrt(np.pi)])

    i = np.arange(1, n)
    a = np.sqrt(i / 2)
    CM = np.diag(a, 1) + np.diag(a, -1)

    eigvals, eigvecs = np.linalg.eigh(CM)
    idx = np.argsort(eigvals)
    x = eigvals[idx]
    V = eigvecs[:, idx]
    w = np.sqrt(np.pi) * V[0, :]**2
    return x, w


def gauss_hermite(n):
    """Compute Gauss-Hermite quadrature nodes and weights via Golub-Welsch.

    Returns nodes x and weights w for integrating integral f(x) * exp(-x^2) dx.
    Nodes are scaled by sqrt(2); weights sum to 1.
    """
    x, w = _gauss_hermite_raw(n)
    x = x * np.sqrt(2)
    w = w / np.sum(w)
    return x, w
