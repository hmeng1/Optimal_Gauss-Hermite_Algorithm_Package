"""Integrand f(x) = prod_i cos^2(x_i) with known closed-form and numerical reference.

Provides:
    func        — f(x) for x of shape (dimX, N), returns (N,)
    true_func   — analytic E[f(X)] for X ~ N(Ex, Cx)
    matlab_func — numerical integration (scipy quad/dblquad/tplquad) for dimX <= 3
"""

import numpy as np
from scipy.linalg import cholesky
from scipy.integrate import quad, dblquad, tplquad

def func(x):
    """f(x) = prod_i cos^2(x_i).  x shape: (dimX, N) -> returns (N,)."""
    return np.prod(np.cos(x) ** 2, axis=0)


def true_func(Ex, Cx):
    """Closed-form E[prod_i cos^2(X_i)] for X ~ N(Ex, Cx).

    Uses the MVN characteristic function:
        cos^2(x) = (e^{2ix} + 2 + e^{-2ix}) / 4
    Expanding the product across dimensions yields a 3^dimX grid of
    frequencies {-2, 0, 2} with tensor-product coefficients.
    """
    dimX = len(Ex)
    vals = np.array([-2, 0, 2])
    grids = np.meshgrid(*[vals] * dimX, indexing='ij')
    gen = np.column_stack([g.ravel() for g in grids])
    coeff = (0.5) ** np.sum(np.abs(gen) == 2, axis=1)
    scale = 0.5 ** dimX
    quadTerm = -0.5 * np.sum((gen @ Cx) * gen, axis=1)
    expo = 1j * (gen @ Ex) + quadTerm
    return np.real(scale * np.sum(coeff * np.exp(expo)))


def matlab_func(Ex, Cx, K=8, RelTol=1e-6, AbsTol=1e-10):
    """Numerical integration of E[prod_i cos^2(X_i)] for dimX = 1, 2, 3."""
    Ex = np.asarray(Ex).ravel()
    d = len(Ex)
    L = cholesky(Cx, lower=True)

    c = 1.0 / ((2 * np.pi) ** (d / 2))

    if d == 1:
        L11 = L[0, 0]
        m1 = Ex[0]
        a = -K if not np.isinf(K) else -np.inf
        b = K if not np.isinf(K) else np.inf

        def f(y1):
            val = np.cos(m1 + L11 * y1) ** 2
            val *= np.exp(-0.5 * y1 ** 2) / np.sqrt(2 * np.pi)
            return val

        q, _ = quad(f, a, b, epsrel=RelTol, epsabs=AbsTol)

    elif d == 2:
        L11 = L[0, 0]
        m1 = Ex[0]
        L21 = L[1, 0]
        L22 = L[1, 1]
        m2 = Ex[1]
        a = -K if not np.isinf(K) else -np.inf
        b = K if not np.isinf(K) else np.inf

        def f(y1, y2):
            val = np.cos(m1 + L11 * y1) ** 2
            val *= np.cos(m2 + L21 * y1 + L22 * y2) ** 2
            val *= c * np.exp(-0.5 * (y1 ** 2 + y2 ** 2))
            return val

        q, _ = dblquad(f, a, b, a, b, epsrel=RelTol, epsabs=AbsTol)

    elif d == 3:
        L11 = L[0, 0]
        m1 = Ex[0]
        L21 = L[1, 0]
        L22 = L[1, 1]
        m2 = Ex[1]
        L31 = L[2, 0]
        L32 = L[2, 1]
        L33 = L[2, 2]
        m3 = Ex[2]
        a = -K if not np.isinf(K) else -np.inf
        b = K if not np.isinf(K) else np.inf

        def f(y1, y2, y3):
            val = np.cos(m1 + L11 * y1) ** 2
            val *= np.cos(m2 + L21 * y1 + L22 * y2) ** 2
            val *= np.cos(m3 + L31 * y1 + L32 * y2 + L33 * y3) ** 2
            val *= c * np.exp(-0.5 * (y1 ** 2 + y2 ** 2 + y3 ** 2))
            return val

        q, _ = tplquad(f, a, b, a, b, a, b, epsrel=RelTol, epsabs=AbsTol)

    else:
        raise ValueError(f"matlab_func supports dimX = 1,2,3 only (got {d}).")

    return q
