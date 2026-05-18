"""Polynomial term matrix construction and left-nullspace basis computation."""

import numpy as np
from pathlib import Path

from ._polynomial_basis import basis_evaluator, poly_terms, poly_terms_full
from .sample_methods import sample_methods
from ._cache_utils import load_pickle_cache, save_pickle_cache


def poly_matrix(num_sample, num_var, lvl, poly_type, use_sparse, sample_type, cache_file=None):
    """Build polynomial term matrix X and orthonormal basis of its left nullspace.

    Returns:
        SampleX (num_var x num_sample): samples
        orgMatrixC ((num_sample - term_num) x num_sample): orthogonal complement rows
    """
    if cache_file is None:
        cache_file = str(Path(__file__).resolve().parent / 'polyMatrixCache.pkl')

    order_poly = 2 * lvl - 1

    meta = {
        'num_sample': num_sample,
        'num_var': num_var,
        'lvl': lvl,
        'poly_type': poly_type,
        'use_sparse': use_sparse,
        'sample_type': sample_type,
    }

    data = load_pickle_cache(cache_file)
    if data is not None and data.get('meta') == meta:
        return data['SampleX'], data['orgMatrixC']

    SampleX = sample_methods(num_var, num_sample, sample_type)
    num_sample = SampleX.shape[1]

    if use_sparse:
        ind = poly_terms(order_poly, num_var)
    else:
        ind = poly_terms_full(order_poly, num_var)

    term_num = ind.shape[0]

    if term_num >= num_sample:
        raise ValueError(
            f"termNum ({term_num}) must be < numSample ({num_sample})")

    eval_basis = basis_evaluator(poly_type, order_poly)

    X = np.ones((num_sample, term_num))
    for v in range(num_var):
        xv = SampleX[v, :]
        Bv = eval_basis(xv)
        X = X * Bv[:, ind[:, v]]

    Q1, _ = np.linalg.qr(X)

    if num_sample <= 5000:
        Qfull, _ = np.linalg.qr(Q1, mode='complete')
        orgMatrixC = Qfull[:, term_num:].T
    else:
        k_null = num_sample - term_num
        # Use a fixed-seed generator for reproducibility
        rng = np.random.default_rng(42)
        Omega = rng.standard_normal((num_sample, min(k_null, 64)))
        Z = Omega - Q1 @ (Q1.T @ Omega)
        Qz, _ = np.linalg.qr(Z)
        Q2 = Qz
        while Q2.shape[1] < k_null:
            Omega = rng.standard_normal((num_sample, min(64, k_null - Q2.shape[1])))
            Z = Omega - Q1 @ (Q1.T @ Omega) - Q2 @ (Q2.T @ Omega)
            Qz, _ = np.linalg.qr(Z)
            Q2 = np.hstack([Q2, Qz])
        Q2 = Q2[:, :k_null]
        orgMatrixC = Q2.T

    data = {'orgMatrixC': orgMatrixC, 'SampleX': SampleX, 'meta': meta}
    save_pickle_cache(cache_file, data)

    return SampleX, orgMatrixC
