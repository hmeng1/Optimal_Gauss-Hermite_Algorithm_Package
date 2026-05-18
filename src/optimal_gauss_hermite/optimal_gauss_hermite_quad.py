"""Main entry point: optimize Gauss-Hermite quadrature parameters for a given integrand."""

import math
import time
import warnings
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from gauss_hermite_quadrature import grid_quadrature
from .poly_matrix import poly_matrix
from .chol.ftrans_chol import ftrans_chol, _pack_chol
from .chol.ftrans_j_chol import ftrans_j_chol
from .ldl.ftrans_ldl import ftrans_ldl, _pack_ldl
from .ldl.ftrans_j_ldl import ftrans_j_ldl
from .optimization_tools.nls_lm_krylov import nls_lm_krylov
from .optimization_tools.nls_lm_exact_h import nls_lm_exact_h
from .optimization_tools.nls_hybrid import nls_hybrid


def optimal_gauss_hermite_quad(
    func: Callable[[np.ndarray], np.ndarray],
    Ex: ArrayLike,
    Cx: ArrayLike,
    sample_num: Optional[int] = None,
    num_quad: int = 3,
    poly_type: str = 'hermite',
    sparse_mode: bool = False,
    sample_type: str = 'latin',
    algorithm_type: str = 'levenberg-marquardt',
    decomp_type: str = 'Chol',
) -> Tuple[float, float]:
    """Optimize Gauss-Hermite quadrature parameters.

    Returns (q_value, elapsed_time).
    """
    dimX = len(Ex)
    Ex = np.asarray(Ex).ravel()
    Cx = np.asarray(Cx)

    if sparse_mode:
        n_coeff = math.comb(dimX + 2 * num_quad - 1, dimX)
    else:
        n_coeff = (2 * num_quad) ** dimX

    if sample_num is None:
        if dimX > 2:
            sample_num = int(np.ceil(1.5 * n_coeff))
        else:
            sample_num = 20 + n_coeff

    if sample_num <= n_coeff:
        warnings.warn(
            f"sample_num ({sample_num}) must be greater than "
            f"the number of polynomial terms n_coeff ({n_coeff}). "
            f"The optimization is underdetermined.",
            stacklevel=2,
        )

    snodes, sweights = grid_quadrature(num_quad, dimX, sparse_mode=sparse_mode)

    SampleX, orgMatrixC = poly_matrix(sample_num, dimX, num_quad,
                                       poly_type, sparse_mode, sample_type)

    if decomp_type.lower() == 'chol':
        x0 = _pack_chol(np.zeros(dimX), np.eye(dimX))

        def resfun(x):
            return ftrans_j_chol(func, x, SampleX, Ex, Cx, orgMatrixC)
    else:
        x0 = _pack_ldl(np.zeros(dimX), np.eye(dimX))

        def resfun(x):
            return ftrans_j_ldl(func, x, SampleX, Ex, Cx, orgMatrixC)

    t0 = time.time()

    algo = algorithm_type.lower()

    if algo == 'levenberg-marquardt':
        opts = {'Display': 'off', 'krylov': 'pcg',
                'mu0': 0.01, 'maxIter': 100}

        def resfun_gn(x):
            return resfun(x)[:2]

        x_opt_sol, _ = nls_lm_krylov(resfun_gn, x0, opts)

    elif algo == 'hybrid':
        opts = {'Display': 'off',
                'mu0': 0.01, 'maxIter': 100,
                'dampingGN': 'diagJTJ',
                'dampingH': 'diagHabs',
                'switchGrad': 1e-5,
                'switchGoodRho': 0.75,
                'switchGoodCount': 2,
                'backRejects': 2,
                'backCholFails': 2}
        x_opt_sol, _ = nls_hybrid(resfun, x0, opts)

    elif algo == 'hessian':
        opts = {'Display': 'off',
                'mu0': 0.01, 'maxIter': 100,
                'damping': 'diagHabs'}
        x_opt_sol, _ = nls_lm_exact_h(resfun, x0, opts)

    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    T = time.time() - t0

    if decomp_type.lower() == 'chol':
        Q = ftrans_chol(func, x_opt_sol, snodes, Ex, Cx)
    else:
        Q = ftrans_ldl(func, x_opt_sol, snodes, Ex, Cx)

    y = sweights @ Q
    return y, T
