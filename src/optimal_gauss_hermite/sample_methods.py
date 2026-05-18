"""Sampling strategies: Sobol, Halton, Latin hypercube, quantile grids."""

import sys
import numpy as np
from scipy.special import erfinv


def sample_methods(num_var, num_sample, sample_type):
    """Generate samples or grid points.

    Returns Sample (num_var x N).
    """
    normal_space = {'latinNorm', 'quantileTensorNorm', 'cubatureNorm', 'unscentedNorm'}

    if sample_type == 'sobol':
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=num_var, scramble=True, seed=1000)
        RandMat = sampler.random(num_sample)

    elif sample_type == 'haltonset':
        from scipy.stats.qmc import Halton
        sampler = Halton(d=num_var, scramble=True, seed=1000)
        RandMat = sampler.random(num_sample)

    elif sample_type == 'latin':
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=num_var)
        RandMat = sampler.random(num_sample)

    elif sample_type == 'MC':
        RandMat = np.random.rand(num_sample, num_var)

    elif sample_type == 'latinNorm':
        from scipy.stats.qmc import LatinHypercube
        from scipy.stats import norm
        sampler = LatinHypercube(d=num_var)
        uniform = sampler.random(num_sample)
        RandMat = norm.ppf(uniform)

    elif sample_type == 'quantile':
        d = num_var
        m = int(np.ceil(np.exp(np.log(num_sample) / d)))
        p = (np.arange(1, m + 1) - 0.5) / m
        q = _norminv_safe(p)

        grids = np.meshgrid(*[q] * d, indexing='ij')
        N = grids[0].size
        RandMat = np.zeros((N, d))
        for k in range(d):
            RandMat[:, k] = grids[k].ravel()

    else:
        raise ValueError(f"Unknown sampling type: {sample_type}")

    if sample_type not in normal_space:
        RandMat = (RandMat - 0.5) * 2.2

    return RandMat.T


def _norminv_safe(p):
    """Normal inverse CDF without Statistics Toolbox."""
    p = np.clip(p, sys.float_info.min, 1.0 - sys.float_info.min)
    return -np.sqrt(2) * erfinv(2 * p)
