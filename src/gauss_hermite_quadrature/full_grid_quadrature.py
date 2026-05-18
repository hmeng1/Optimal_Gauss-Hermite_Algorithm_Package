"""Full tensor-product Gauss-Hermite quadrature."""

import numpy as np
from itertools import product

from .gauss_hermite import gauss_hermite


def full_grid_quadrature(num_quad, num_var):
    """Full tensor-product Gauss-Hermite quadrature.

    Returns nodes (num_var x N) and weights (N,).
    """
    qnodes, qweight = gauss_hermite(num_quad)

    idx_grid = np.array(list(product(range(num_quad), repeat=num_var)))
    nodes = qnodes[idx_grid]

    nodes = np.where(np.abs(nodes) > 1e-10, nodes, 0.0)

    if num_var == 1:
        weights = qweight[idx_grid[:, 0]]
    else:
        weights = np.prod(qweight[idx_grid], axis=1)

    return nodes.T, weights
