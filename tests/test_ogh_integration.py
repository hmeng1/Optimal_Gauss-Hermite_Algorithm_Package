import numpy as np
from ogh_integration import ogh_integrate


def test_ogh_integrate_basic():
    np.random.seed(42)
    dimX = 2
    Ex = np.random.randn(dimX)
    Csqrt = np.random.randn(dimX, dimX)
    Cx = Csqrt @ Csqrt.T

    def f(x):
        return np.prod(np.cos(x)**2, axis=0)

    result = ogh_integrate(f, Ex, Cx, num_quad=2)
    assert np.isfinite(result)
    assert isinstance(result, float)


def test_ogh_integrate_constant():
    Ex = np.array([1.0, 2.0])
    Cx = np.array([[1.0, 0.0], [0.0, 1.0]])

    def f_constant(x):
        return np.ones(x.shape[1])

    result = ogh_integrate(f_constant, Ex, Cx, num_quad=3)
    # quadrature is approximate; constant integrand should be close to 1
    assert abs(result - 1.0) < 0.05


def test_ogh_integrate_sparse_mode():
    np.random.seed(123)
    Ex = np.array([0.5, -0.1])
    Cx = np.array([[2.0, 0.5], [0.5, 1.0]])

    def f(x):
        return np.exp(-np.sum(x**2, axis=0) / 4)

    result = ogh_integrate(f, Ex, Cx, num_quad=3, sparse_mode=True, sample_num=30)
    assert np.isfinite(result)
