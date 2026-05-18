import numpy as np
from gauss_hermite_quadrature import gauss_hermite, full_grid_quadrature, grid_quadrature


def test_gauss_hermite_nodes_weights_shape():
    n = 5
    x, w = gauss_hermite(n)
    assert x.shape == (n,)
    assert w.shape == (n,)


def test_gauss_hermite_weights_sum_to_one():
    for n in [1, 3, 5, 10]:
        _, w = gauss_hermite(n)
        assert abs(np.sum(w) - 1.0) < 1e-12


def test_gauss_hermite_integrates_polynomials():
    for n in range(2, 6):
        x, w = gauss_hermite(n)
        # weights sum to 1
        assert abs(np.sum(w) - 1.0) < 1e-12
        # raw nodes (x/sqrt(2)) and unnormalized weights integrate x^0 -> sqrt(pi)
        x_raw = x / np.sqrt(2)
        w_raw = w * np.sqrt(np.pi)
        integral_const = np.sum(w_raw * np.ones_like(x_raw))
        assert abs(integral_const - np.sqrt(np.pi)) < 1e-12
        # x^2 * exp(-x^2) integrates to sqrt(pi)/2
        integral_x2 = np.sum(w_raw * x_raw**2)
        assert abs(integral_x2 - np.sqrt(np.pi) / 2) < 1e-10


def test_full_grid_quadrature_shape():
    lvl, dim = 3, 2
    nodes, weights = full_grid_quadrature(lvl, dim)
    assert nodes.shape[0] == dim
    assert nodes.shape[1] == lvl ** dim
    assert weights.shape[0] == lvl ** dim


def test_full_grid_weights_sum_to_one():
    lvl, dim = 3, 2
    _, weights = full_grid_quadrature(lvl, dim)
    assert abs(np.sum(weights) - 1.0) < 1e-12


def test_grid_quadrature_dispatch_full():
    nodes, weights = grid_quadrature(3, 2, sparse_mode=False)
    nodes2, weights2 = full_grid_quadrature(3, 2)
    assert np.allclose(nodes, nodes2)
    assert np.allclose(weights, weights2)


def test_grid_quadrature_dispatch_sparse():
    nodes, weights = grid_quadrature(3, 2, sparse_mode=True)
    assert nodes.shape[0] == 2
    assert weights.ndim == 1
    assert abs(np.sum(weights) - 1.0) < 1e-12
