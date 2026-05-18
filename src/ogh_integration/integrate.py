"""Compute E[f(X)] for X ~ N(Ex, Cx) using optimized Gauss-Hermite quadrature."""

from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from optimal_gauss_hermite.optimal_gauss_hermite_quad import (
    optimal_gauss_hermite_quad,
)


def ogh_integrate(
    func: Callable[[np.ndarray], np.ndarray],
    Ex: ArrayLike,
    Cx: ArrayLike,
    *,  # everything below is keyword-only
    sample_num: Optional[int] = None,
    num_quad: int = 3,
    poly_type: str = "hermite",
    sparse_mode: bool = False,
    sample_type: str = "latin",
    algorithm_type: str = "levenberg-marquardt",
    decomp_type: str = "Chol",
) -> float:
    """Compute E[f(X)] for X ~ N(Ex, Cx) via optimal Gauss-Hermite quadrature.

    Parameters
    ----------
    func : callable
        Integrand f(X) where X is (dimX, N) and returns (N,) or (1, N).
    Ex : array_like (dimX,)
        Mean of the multivariate normal distribution.
    Cx : array_like (dimX, dimX)
        Covariance matrix (must be symmetric positive-definite).
    sample_num : int, optional
        Number of sample points for optimization. Defaults to
        ceil(1.5 * (2*lvl)^dimX) when dimX > 2, else 20.
    num_quad : int
        Quadrature level (default 3).
    poly_type : str
        Polynomial basis type: 'hermite', 'chebyshev1st', 'chebyshev2nd',
        'laguerre', 'canonical'.
    sparse_mode : bool
        Use sparse-grid quadrature if True.
    sample_type : str
        Sampling method for optimization: 'latin', 'sobol', 'haltonset', 'MC'.
    algorithm_type : str
        Optimization algorithm: 'levenberg-marquardt', 'hybrid', 'hessian'.
    decomp_type : str
        Parameterization: 'Chol' or 'LDL'.

    Returns
    -------
    float
        The estimated integral E[f(X)].
    """
    Ex = np.asarray(Ex, dtype=float).ravel()
    Cx = np.asarray(Cx, dtype=float)

    result, elapsed = optimal_gauss_hermite_quad(
        func,
        Ex,
        Cx,
        sample_num=sample_num,
        num_quad=num_quad,
        poly_type=poly_type,
        sparse_mode=sparse_mode,
        sample_type=sample_type,
        algorithm_type=algorithm_type,
        decomp_type=decomp_type,
    )

    return result
