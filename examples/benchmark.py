"""Monte Carlo benchmark comparing quadrature methods for E[f(X)] with X ~ N(Ex, Cx).

The integrand must be supplied via `func`, with optional analytic truth
(`true_func`) and numerical reference (`matlab_func`). Rows for which no
reference is available are automatically omitted from the display.
"""

import sys
import time
from pathlib import Path
import hashlib
import pickle
import numpy as np
from math import comb
from scipy.linalg import cholesky

from gauss_hermite_quadrature import grid_quadrature
from optimal_gauss_hermite import optimal_gauss_hermite_quad
from _display import print_run_table, print_summary
from optimal_gauss_hermite._cache_utils import load_pickle_cache, save_pickle_cache


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def local_hash(obj):
    """MD5 hex hash of a Python object via pickle."""
    bs = pickle.dumps(obj, protocol=4)
    return hashlib.md5(bs).hexdigest()


def _func_id(func):
    """Best-effort string identifier for a callable (for cache-key use)."""
    try:
        name = func.__name__
    except AttributeError:
        name = 'unknown'
    try:
        code_hash = hashlib.md5(func.__code__.co_code).hexdigest()[:8]
        return f"{name}_{code_hash}"
    except (AttributeError, TypeError):
        return name


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main(dimX=3, lvl=3, func=None, true_function=None, matlab_function=None,
         algorithm_name='levenberg-marquardt', sparse=False,
         sample_type='latin', decomp_type='Chol'):
    """Benchmark Gauss-Hermite quadrature methods for E[f(X)] with X ~ N(Ex, Cx).

    Parameters
    ----------
    dimX : int
        Dimension of the multivariate normal.
    lvl : int
        Quadrature level (number of nodes per dimension in each direction).
    func : callable
        Integrand f(x) where x has shape (dimX, N). Must return (N,) or (1, N).
    true_function : callable(Ex, Cx) -> float or None
        Analytic expected value E[f(X)]. If None the 'True' row is omitted.
    matlab_func : callable(Ex, Cx) -> float or None
        Numerical integration reference. If None the 'Matlab' row is omitted.
    algorithm_name : str
        'levenberg-marquardt' | 'hybrid' | 'hessian'.
    sparse : bool
        Use sparse-grid quadrature.
    sample_type : str
        'latin' | 'sobol' | 'haltonset' | 'MC' | 'latinNorm' | 'quantile'.
    decomp_type : str
        'Chol' | 'LDL'.
    """
    cache_path = Path(__file__).resolve().parent.parent / 'quadratureCache.pkl'
    cache_file = str(cache_path)
    use_cache = True
    clear_cache = True

    if clear_cache and cache_path.is_file():
        cache_path.unlink()

    isSparse = sparse
    algorithmName = algorithm_name
    decompType = decomp_type

    numRuns = 100
    verbose = True

    has_true = true_function is not None
    has_matlab = matlab_function is not None

    # --- Build row names dynamically ---
    row_names = []
    if has_true:
        row_names.append('True')
    if has_matlab:
        row_names.append('Matlab')
    row_names.extend(['GHQ', 'Optimized GHQ'])

    n_methods = len(row_names)
    idx_ghq = row_names.index('GHQ')
    idx_oghq = row_names.index('Optimized GHQ')

    nUnknown = dimX * (dimX + 3) // 2

    if not isSparse:
        nCoeffLinear = (2 * lvl) ** dimX
    else:
        nCoeffLinear = comb(dimX + 2 * lvl - 1, dimX)

    if dimX > 2:
        numSample = int(np.ceil(1.5 * nCoeffLinear))
    else:
        numSample = 20

    print(f"Integrand: {_func_id(func)}")
    print(f"Unknowns: {nUnknown}")
    print(f"Linear coefficients: {nCoeffLinear}")

    # --- Allocate result arrays ---
    Q_all = np.full((n_methods, numRuns), np.nan)
    err_all = np.full((n_methods, numRuns), np.nan)
    t_all = np.full((n_methods, numRuns), np.nan)
    better = np.zeros(numRuns, dtype=bool)

    # --- Quadrature nodes ---
    gh_nodes, w = grid_quadrature(lvl, dimX, sparse_mode=isSparse)

    # --- Cache setup ---
    if use_cache:
        cache = load_pickle_cache(cache_file, default={'keys': [], 'values': []})
        if not isinstance(cache, dict) or 'keys' not in cache:
            cache = {'keys': [], 'values': []}

    # --- Main loop ---
    for runIdx in range(1, numRuns + 1):
        np.random.seed(runIdx + 10000)

        Ex = np.random.randn(dimX)
        Csqrt = np.random.randn(dimX, dimX)
        Cx = Csqrt @ Csqrt.T
        Lc = cholesky(Cx, lower=True)

        if verbose:
            print(f"\nRun {runIdx} / {numRuns}")

        ref_val = None  # reference value for error computation

        # --- True (analytic closed-form) ---
        if has_true:
            q_true = true_function(Ex, Cx)
            Q_all[0, runIdx - 1] = q_true
            err_all[0, runIdx - 1] = 0.0
            ref_val = q_true

        # --- Matlab (numerical reference) ---
        if has_matlab:
            t0 = time.time()
            q_matlab = matlab_function(Ex, Cx)
            t_matlab = time.time() - t0
            mlab_col = 1 if has_true else 0
            Q_all[mlab_col, runIdx - 1] = q_matlab
            t_all[mlab_col, runIdx - 1] = t_matlab
            if ref_val is None:
                ref_val = q_matlab

        # --- GHQ ---
        qx = Ex[:, np.newaxis] + Lc @ gh_nodes
        t0 = time.time()
        q_ghq = w @ func(qx)
        t_ghq = time.time() - t0
        Q_all[idx_ghq, runIdx - 1] = q_ghq
        t_all[idx_ghq, runIdx - 1] = t_ghq
        if ref_val is None:
            ref_val = q_ghq

        # --- Optimized GHQ ---
        if use_cache:
            key_data = (_func_id(func), tuple(Ex), tuple(Cx.ravel()),
                        numSample, lvl, isSparse, algorithmName,
                        decompType, 'hermite', sample_type)
            key = local_hash(key_data)

            if key in cache['keys']:
                idx = cache['keys'].index(key)
                q_oghq = cache['values'][idx]
                t_oghq = np.nan
            else:
                q_oghq, t_oghq = optimal_gauss_hermite_quad(
                    func, Ex, Cx,
                    sample_num=numSample,
                    num_quad=lvl,
                    poly_type='hermite',
                    sparse_mode=isSparse,
                    sample_type=sample_type,
                    algorithm_type=algorithmName,
                    decomp_type=decompType)
                cache['keys'].append(key)
                cache['values'].append(q_oghq)
                save_pickle_cache(cache_file, cache)
        else:
            q_oghq, t_oghq = optimal_gauss_hermite_quad(
                func, Ex, Cx,
                sample_num=numSample,
                num_quad=lvl,
                poly_type='hermite',
                sparse_mode=isSparse,
                sample_type=sample_type,
                algorithm_type=algorithmName,
                decomp_type=decompType)

        Q_all[idx_oghq, runIdx - 1] = q_oghq
        t_all[idx_oghq, runIdx - 1] = t_oghq

        # --- Error computation (relative to best available reference) ---
        denom = max(sys.float_info.epsilon, abs(ref_val))
        for i in range(n_methods):
            if np.isfinite(Q_all[i, runIdx - 1]):
                err_all[i, runIdx - 1] = (
                    np.abs(Q_all[i, runIdx - 1] - ref_val) / denom * 100)
        if has_true:
            err_all[0, runIdx - 1] = 0.0

        better[runIdx - 1] = (
            err_all[idx_oghq, runIdx - 1] <= err_all[idx_ghq, runIdx - 1])

        # --- Per-run display ---
        if verbose:
            Q_row = Q_all[:, runIdx - 1]
            err_row = err_all[:, runIdx - 1]
            t_ghq_val = t_all[idx_ghq, runIdx - 1]
            denom_t = max(t_ghq_val, sys.float_info.epsilon)
            time_ratio = np.array([
                t_all[i, runIdx - 1] / denom_t for i in range(n_methods)
            ])
            print_run_table(row_names, Q_row, err_row, time_ratio)
            print(f"Better fraction so far: {np.mean(better[:runIdx]):.4f}")

    # --- Summary ---
    betterFrac = np.mean(better)
    print_summary(row_names, err_all, t_all, betterFrac)


if __name__ == '__main__':
    import _cos2_integrand as _integrand

    main(dimX=3, lvl=3, func=_integrand.func,
         true_function=getattr(_integrand, 'true_func', None),
         matlab_function=getattr(_integrand, 'matlab_func', None))
