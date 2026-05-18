"""Dispatcher selecting full-grid or sparse-grid Gauss-Hermite quadrature."""

from .full_grid_quadrature import full_grid_quadrature
from .sparse_grid import sg_gh_gaussian_sparsegrid


def grid_quadrature(num_quad, num_var, sparse_mode=False):
    """Dispatch to full-grid or sparse-grid Gauss-Hermite quadrature.

    Returns nodes (num_var x N) and weights (N,).
    """
    if not sparse_mode or num_var == 1:
        return full_grid_quadrature(num_quad, num_var)
    else:
        return sg_gh_gaussian_sparsegrid(num_var, num_quad)
