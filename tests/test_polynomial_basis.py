import numpy as np
from optimal_gauss_hermite._polynomial_basis import (
    basis_evaluator,
    poly_terms,
    poly_terms_full,
)


def test_basis_evaluator_hermite_shape():
    eval_basis = basis_evaluator('hermite', 5)
    x = np.array([0.0, 1.0, -1.0])
    B = eval_basis(x)
    assert B.shape == (3, 6)  # n_points x (order + 1)


def test_basis_evaluator_hermite_values():
    eval_basis = basis_evaluator('hermite', 5)
    x = np.array([0.0])
    B = eval_basis(x)
    # He_0(0) = 1, He_1(0) = 0, He_2(0) = -1, He_3(0) = 0, He_4(0) = 3, He_5(0) = 0
    expected = np.array([[1.0, 0.0, -1.0, 0.0, 3.0, 0.0]])
    assert np.allclose(B, expected)


def test_basis_evaluator_canonical():
    eval_basis = basis_evaluator('canonical', 3)
    x = np.array([2.0])
    B = eval_basis(x)
    expected = np.array([[1.0, 2.0, 4.0, 8.0]])
    assert np.allclose(B, expected)


def test_poly_terms_shape_total_degree():
    ind = poly_terms(3, 2)
    # number of multi-indices with sum <= 3 in 2 vars = C(3+2, 3) = 10
    assert ind.shape == (10, 2)


def test_poly_terms_full_shape():
    ind = poly_terms_full(3, 2)
    # full grid: (3+1)^2 = 16
    assert ind.shape == (16, 2)


def test_basis_evaluator_chebyshev1st():
    eval_basis = basis_evaluator('chebyshev1st', 3)
    x = np.array([0.5])
    B = eval_basis(x)
    # T0=1, T1=0.5, T2=2*0.25-1=-0.5, T3=2*0.5*(-0.5)-0.5=-1.0
    assert abs(B[0, 0] - 1.0) < 1e-12
    assert abs(B[0, 1] - 0.5) < 1e-12
    assert abs(B[0, 2] + 0.5) < 1e-12
    assert abs(B[0, 3] + 1.0) < 1e-12


def test_basis_evaluator_laguerre():
    eval_basis = basis_evaluator('laguerre', 2)
    x = np.array([0.0])
    B = eval_basis(x)
    # L0=1, L1=1, L2=1
    assert np.allclose(B[0, :], [1.0, 1.0, 1.0])
