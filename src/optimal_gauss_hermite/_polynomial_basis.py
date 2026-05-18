"""Polynomial basis evaluators, coefficient generators, and multi-index helpers."""

import numpy as np
from math import comb
from itertools import combinations


# ---------------------------------------------------------------------------
# Multi-index generators
# ---------------------------------------------------------------------------

def poly_terms(order_poly, num_var):
    """Total-degree multi-indices alpha >= 0 with sum(alpha) <= order_poly."""
    n_rows = comb(order_poly + num_var, order_poly)
    ind = np.zeros((n_rows, num_var), dtype=np.int32)
    row = 0
    ind[row, :] = 0
    row += 1

    for s in range(1, order_poly + 1):
        comps = _weak_compositions(s, num_var)
        add = comps.shape[0]
        ind[row:row + add, :] = comps
        row += add
    return ind


def poly_terms_full(order_poly, num_var):
    """Full tensor grid multi-indices in {0..order_poly}^num_var."""
    vec = np.arange(order_poly + 1)
    grids = np.meshgrid(*[vec] * num_var, indexing='ij')
    cols = [g.ravel() for g in grids]
    return np.column_stack(cols).astype(np.int32)


def _weak_compositions(s, m):
    """All weak compositions of integer s into m parts (>=0)."""
    if m == 1:
        return np.array([[s]], dtype=np.int32)

    bars = np.array(list(combinations(range(1, s + m), m - 1)), dtype=np.int32)
    n = bars.shape[0]
    padded = np.hstack([np.zeros((n, 1), dtype=np.int32), bars,
                        np.full((n, 1), s + m, dtype=np.int32)])
    C = np.diff(padded, axis=1) - 1
    return C.astype(np.int32)


# ---------------------------------------------------------------------------
# Basis evaluator factory
# ---------------------------------------------------------------------------

def basis_evaluator(poly_type, order_poly):
    """Return a function that evaluates all basis polynomials up to order_poly."""
    pt = poly_type.lower()
    if pt == 'canonical':
        return lambda x: _eval_canonical(x, order_poly)
    elif pt == 'chebyshev1st':
        return lambda x: _eval_cheb_t(x, order_poly)
    elif pt == 'chebyshev2nd':
        return lambda x: _eval_cheb_u(x, order_poly)
    elif pt == 'hermite':
        return lambda x: _eval_hermite_prob(x, order_poly)
    elif pt == 'laguerre':
        return lambda x: _eval_laguerre(x, order_poly)
    else:
        basis_coef = poly_basis_coef(order_poly, poly_type)
        return lambda x: _eval_polyval_stack(x, basis_coef, order_poly)


# ---------------------------------------------------------------------------
# Direct recurrence evaluators (fast path for known polynomial families)
# ---------------------------------------------------------------------------

def _eval_canonical(x, p):
    """B[:, k] = x^k for k = 0..p."""
    n = len(x)
    B = np.ones((n, p + 1))
    if p == 0:
        return B
    B[:, 1] = x
    for k in range(2, p + 1):
        B[:, k] = B[:, k - 1] * x
    return B


def _eval_cheb_t(x, p):
    """Chebyshev 1st kind: T0=1, T1=x, T_{k+1}=2x T_k - T_{k-1}."""
    n = len(x)
    B = np.zeros((n, p + 1))
    B[:, 0] = 1
    if p == 0:
        return B
    B[:, 1] = x
    for k in range(2, p + 1):
        B[:, k] = 2 * x * B[:, k - 1] - B[:, k - 2]
    return B


def _eval_cheb_u(x, p):
    """Chebyshev 2nd kind: U0=1, U1=2x, U_{k+1}=2x U_k - U_{k-1}."""
    n = len(x)
    B = np.zeros((n, p + 1))
    B[:, 0] = 1
    if p == 0:
        return B
    B[:, 1] = 2 * x
    for k in range(2, p + 1):
        B[:, k] = 2 * x * B[:, k - 1] - B[:, k - 2]
    return B


def _eval_hermite_prob(x, p):
    """Probabilists' Hermite: He0=1, He1=x, He_{k+1}=x He_k - k He_{k-1}."""
    n = len(x)
    B = np.zeros((n, p + 1))
    B[:, 0] = 1
    if p == 0:
        return B
    B[:, 1] = x
    for k in range(1, p):
        B[:, k + 1] = x * B[:, k] - k * B[:, k - 1]
    return B


def _eval_laguerre(x, p):
    """Laguerre: L0=1, L1=1-x, (k+1)L_{k+1}=(2k+1-x)L_k - k L_{k-1}."""
    n = len(x)
    B = np.zeros((n, p + 1))
    B[:, 0] = 1
    if p == 0:
        return B
    B[:, 1] = 1 - x
    for k in range(1, p):
        B[:, k + 1] = ((2 * k + 1 - x) * B[:, k] - k * B[:, k - 1]) / (k + 1)
    return B


# ---------------------------------------------------------------------------
# Coefficient generators (for custom polynomial types)
# ---------------------------------------------------------------------------

def poly_basis_coef(order_poly, poly_type):
    """Polynomial coefficients up to order_poly (excluding constant term)."""
    y = np.zeros((order_poly, order_poly + 1))
    pt = poly_type.lower()
    for i in range(1, order_poly + 1):
        if pt == 'canonical':
            y[i - 1, -(i + 1)] = 1
        elif pt == 'hermite':
            coeffs = _hermite_rec(i)
            y[i - 1, -len(coeffs):] = coeffs
        elif pt == 'chebyshev1st':
            coeffs = _chebyshev1st_rec(i)
            y[i - 1, -len(coeffs):] = coeffs
        elif pt == 'chebyshev2nd':
            coeffs = _chebyshev2nd_rec(i)
            y[i - 1, -len(coeffs):] = coeffs
        elif pt == 'laguerre':
            coeffs = _laguerre_rec(i)
            y[i - 1, -len(coeffs):] = coeffs
        else:
            raise ValueError(f"Unknown polyType: {poly_type}")
    return y


def _hermite_rec(n):
    """Probabilists' Hermite polynomial coefficients for degree n."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([1.0, 0.0])
    H_prev = np.array([1.0, 0.0])
    H_prev2 = np.array([1.0])
    for k in range(2, n + 1):
        H = np.zeros(k + 1)
        H[0] = 1.0
        H[1:] = H_prev[:-1] - (k - 1) * H_prev2
        H_prev2 = H_prev
        H_prev = H
    return H_prev


def _chebyshev1st_rec(n):
    """Chebyshev 1st kind coefficients for degree n."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([1.0, 0.0])
    T_prev = np.array([1.0, 0.0])
    T_prev2 = np.array([1.0])
    for k in range(2, n + 1):
        T = np.zeros(k + 1)
        T[:-1] = 2 * T_prev[:-1]
        T -= T_prev2
        T_prev2 = T_prev
        T_prev = T
    return T_prev


def _chebyshev2nd_rec(n):
    """Chebyshev 2nd kind coefficients for degree n."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    U_prev = np.array([2.0, 0.0])
    U_prev2 = np.array([1.0])
    for k in range(2, n + 1):
        U = np.zeros(k + 1)
        U[:-1] = 2 * U_prev[:-1]
        U -= U_prev2
        U_prev2 = U_prev
        U_prev = U
    return U_prev


def _laguerre_rec(n):
    """Laguerre polynomial coefficients for degree n."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([-1.0, 1.0])
    L_prev = np.array([-1.0, 1.0])
    L_prev2 = np.array([1.0])
    for k in range(1, n):
        L = np.zeros(k + 2)
        term1 = (2 * k + 1) / (k + 1) * L_prev
        term2 = k / (k + 1) * L_prev2
        L[-len(term1):] = term1
        L[-len(term2):] -= term2
        L[1:] -= np.append(np.array([0.0]), L_prev[:-1] / (k + 1))
        L_prev2 = L_prev
        L_prev = L
    return L_prev


# ---------------------------------------------------------------------------
# NumPy polyval fallback
# ---------------------------------------------------------------------------

def _eval_polyval_stack(x, basis_coef, order_poly):
    """Fallback: evaluate all degrees via np.polyval."""
    n = len(x)
    B = np.ones((n, order_poly + 1))
    for k in range(1, order_poly + 1):
        ck = basis_coef[k - 1, :]
        B[:, k] = np.polyval(ck, x)
    return B
