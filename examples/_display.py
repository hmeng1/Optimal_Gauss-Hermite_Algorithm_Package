"""Formatting helpers for quadrature benchmark output."""

import warnings
import numpy as np


def _nanmedian_safe(a, axis=None):
    """np.nanmedian without the All-NaN slice warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def fmt_num(x, w):
    if not np.isfinite(x):
        return f"{' ':{w}}"
    ax = abs(x)
    if ax == 0 or (ax >= 1e-6 and ax < 1e6):
        return f"{x:{w}.6f}"
    return f"{x:{w}.6e}"


def fmt_pct(x, w):
    if not np.isfinite(x):
        return f"{' ':{w}}"
    return f"{x:{w}.4f}"


def print_run_table(row_names, Q, err_pct, time_vs_ghq):
    nameW = max(max(len(n) for n in row_names), len("Method"))
    print()
    print(f"{'Method':<{nameW}}  {'Result':>14}  {'Err(%)':>9}  {'Time/GHQ':>10}")
    print('-' * (nameW + 2 + 14 + 2 + 9 + 2 + 10))
    for i, name in enumerate(row_names):
        print(f"{name:<{nameW}}  {fmt_num(Q[i], 14)}  {fmt_pct(err_pct[i], 9)}  {fmt_num(time_vs_ghq[i], 10)}")


def print_summary(row_names, err_all, t_all, better_frac):
    """Print benchmark summary.

    Parameters
    ----------
    row_names : list of str
    err_all : ndarray (n_methods, n_runs)
    t_all : ndarray (n_methods, n_runs) — NaN for non-timed methods (e.g., True).
    better_frac : float
    """
    n_methods, numRuns = err_all.shape

    medErr = _nanmedian_safe(err_all, axis=1)
    medTime = _nanmedian_safe(t_all, axis=1)

    ghq_idx = row_names.index('GHQ')
    denom = t_all[ghq_idx, :].copy()
    denom[denom <= 0] = np.nan
    denom[~np.isfinite(denom)] = np.nan

    medTRatio = np.full(n_methods, np.nan)
    for i in range(n_methods):
        ratios = t_all[i, :] / denom
        medTRatio[i] = _nanmedian_safe(ratios)

    print(f"\n{'=' * 20} SUMMARY ({numRuns} runs) {'=' * 20}")
    print(f"Better fraction (Optimized < GHQ): {better_frac:.4f}")

    nameW = max(max(len(n) for n in row_names), len("Method"))
    print()
    print(f"{'Method':<{nameW}}  {'MedErr(%)':>9}  {'MedTime(s)':>12}  {'MedTime/GHQ':>10}")
    print('-' * (nameW + 2 + 9 + 2 + 12 + 2 + 10))
    for i, name in enumerate(row_names):
        print(f"{name:<{nameW}}  {fmt_pct(medErr[i], 9)}  {fmt_num(medTime[i], 12)}  {fmt_num(medTRatio[i], 10)}")
