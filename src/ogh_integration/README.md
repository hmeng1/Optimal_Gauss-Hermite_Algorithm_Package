# Optimal Gauss-Hermite Quadrature Integration

Compute E[f(X)] for X ~ N(Ex, Cx) using optimized Gauss-Hermite quadrature.

## Quick start

```bash
python examples/ogh_integrate_demo.py
```

## Use in your own code

```python
import numpy as np
from ogh_integration import ogh_integrate

Ex = np.array([0.5, -0.1])
Cx = np.array([[2.0, 0.5],
               [0.5, 1.0]])

def f(x):
    # f receives x of shape (dimX, N) and must return (N,)
    return np.prod(np.cos(x)**2, axis=0)

result = ogh_integrate(f, Ex, Cx)
print(f"E[f(X)] = {result:.12f}")
```

## API

```python
ogh_integrate(
    func,           # f(X) with X (dimX, N) → (N,)
    Ex,             # mean (dimX,)
    Cx,             # covariance (dimX, dimX), SPD
    *,              # all remaining args must be named
    sample_num=None,        # sample points for optimization (auto if None)
    num_quad=3,             # quadrature level
    sparse_mode=False,      # True for sparse-grid quadrature
    poly_type="hermite",    # polynomial basis
    sample_type="latin",    # sampling: "latin", "sobol", "haltonset", "MC"
    algorithm_type="levenberg-marquardt",  # "hybrid" or "hessian"
    decomp_type="Chol",     # "Chol" or "LDL"
)
```

Tune `num_quad` for accuracy — larger values use more quadrature points. Use `sparse_mode=True` for high-dimensional problems to avoid exponential growth.
