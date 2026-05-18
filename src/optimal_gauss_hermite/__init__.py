"""Optimized Gauss-Hermite quadrature with adaptive parameter optimization."""

from .optimal_gauss_hermite_quad import optimal_gauss_hermite_quad
from .poly_matrix import poly_matrix
from .sample_methods import sample_methods

__all__ = ['optimal_gauss_hermite_quad', 'poly_matrix', 'sample_methods']
